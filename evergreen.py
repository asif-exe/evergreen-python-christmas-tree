from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# -------------------------- Scene / camera --------------------------
W, H = 600, 1066
SCENE_H = 880
CX, FLOOR, TOP = W // 2, 809, 125
TREE_H = FLOOR - TOP
FPS = 24
TAU = math.tau
COLORS = np.array([
    (255, 210, 110), (255, 235, 175), (247, 103, 69),
    (112, 218, 255), (255, 150, 187), (255, 185, 66),
], dtype=np.float32)  # RGB


def clamp(x, a=0.0, b=1.0):
    return np.clip(x, a, b)


def smooth(t):
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def radius_at(h):
    return 265.0 * np.maximum(0.004, 1.0 - h) ** 0.91


def stars_8(x: float, y: float, r: float, phase: float):
    # Precision-cut jewel, alternating long and short rays.
    pts=[]
    for i in range(16):
        a=phase + i*TAU/16
        d=r*(1.0 if i%2==0 else 0.36)
        pts.append((int(round(x+math.cos(a)*d)),int(round(y+math.sin(a)*d))))
    return np.array(pts,np.int32)


GLYPHS = {
    'v': ('00000','00000','10001','01010','00100','00000'),
    'w': ('00000','10001','10001','10101','01010','00000'),
    'Y': ('10001','01010','00100','00100','00100','00000'),
    'x': ('00000','10001','01010','00100','01010','10001'),
    'A': ('00100','01010','10001','11111','10001','00000'),
    'm': ('00000','00000','11010','10101','10101','00000'),
    '#': ('01010','11111','01010','11111','01010','00000'),
    '*': ('00000','10101','01110','11111','01110','10101'),
    '+': ('00000','00100','00100','11111','00100','00100'),
    '/': ('00001','00010','00100','01000','10000','00000'),
    '%': ('11001','11010','00100','01011','10011','00000'),
}
CHARSET=list(GLYPHS)
GLYPH_MASKS={key:np.array([(xx,yy) for yy,row in enumerate(pattern) for xx,ch in enumerate(row) if ch=='1'],np.int16)
             for key,pattern in GLYPHS.items()}


@dataclass
class Scene3D:
    pos: np.ndarray
    shade: np.ndarray
    kind: np.ndarray
    glyph: np.ndarray
    seed: np.ndarray
    twig_segments: np.ndarray
    tree_light_idx: np.ndarray
    tree_light_strength: np.ndarray
    bulbs: np.ndarray
    bulb_colors: np.ndarray
    bulb_phase: np.ndarray
    garland: np.ndarray
    ornaments: np.ndarray
    ornament_colors: np.ndarray
    ornament_phase: np.ndarray
    gifts: list


def build_geometry(seed=2906) -> Scene3D:
    rng=np.random.default_rng(seed)
    points=[]; shades=[]; kinds=[]; glyphs=[]; seeds=[]; segments=[]

    def needles(center, direction, count, color_kind=0, spread=6.5):
        # short needle fans around actual 3D boughs
        for _ in range(count):
            a=rng.uniform(-.73,.73)
            d=rng.uniform(-spread, spread)
            x=center[0]+math.cos(direction+a)*d
            z=center[2]+math.sin(direction+a)*d
            y=center[1]+rng.uniform(-.012,.012)
            points.append((x,y,z))
            shades.append(rng.uniform(.22,.98))
            kinds.append(color_kind)
            glyphs.append(rng.choice([0,1,2,3,4,5,6,7,8,9,10],p=[.22,.18,.12,.10,.08,.08,.04,.07,.04,.04,.03]))
            seeds.append(rng.random())

    # Oppositely staggered branch whorls, with sagging droplet-shaped boughs.
    for tier in range(24):
        h=.034 + tier*.0406
        if h>=.99:break
        base_width=float(radius_at(h))
        n_branches=int(9 + (1-h)*5)
        whorl_rotation=tier*2.39996
        for b in range(n_branches):
            az=(b+0.16*rng.uniform(-1,1))*TAU/n_branches+whorl_rotation
            arm=base_width*rng.uniform(.69,1.20)
            drop=.009+.031*(1-h)
            spine_pts=[]
            for k in range(21):
                f=k/20
                sway=6.5*math.sin(f*6.3+tier*.3+b*.7)*f
                r=arm*f
                x=math.cos(az)*r - math.sin(az)*sway
                z=math.sin(az)*r + math.cos(az)*sway
                y=h-drop*f**1.72+0.006*math.sin(f*4+az)*f
                loc=(x,y,z); spine_pts.append(loc)
                needles(loc,az,2 if k<17 else 4,0, 3+5*(1-f))

                # Alternating lateral branchlets, densely feathered at tips.
                if 3<=k<=17 and k%3==0:
                    for side in (-1,1):
                        extent=(.20+.18*rng.random())*arm*(.35+.65*(1-f))
                        ang=az+side*(.50+.12*rng.random())
                        start=np.array(loc)
                        last=start.copy()
                        for q in range(8):
                            ff=(q+1)/8
                            bend=extent*ff
                            child=np.array((x+math.cos(ang)*bend,
                                            y-0.015*ff**1.3+0.008*math.sin(ff*4),
                                            z+math.sin(ang)*bend))
                            needles(child,ang,1 if q<6 else 3,1, 3.7)
                            if q%2==0:segments.append((last,child))
                            last=child
            # dark wooden branch spines recede behind glyphs
            for k in range(0,len(spine_pts)-1,2):
                segments.append((spine_pts[k],spine_pts[min(k+2,len(spine_pts)-1)]))

            # Individual trailing tips keep an organic silhouette.
            for k in range(11):
                f=.88+.014*k
                x=math.cos(az)*arm*f
                z=math.sin(az)*arm*f
                y=h-drop*f**1.72-.003*k
                needles((x,y,z),az,2,2,3.2)

    # Dark interior needle volume, leaving whorl gaps around the silhouette.
    for i in range(3000):
        h=rng.uniform(.035,.98)
        theta=rng.uniform(0,TAU)
        rad=float(radius_at(h))*math.sqrt(rng.random())*rng.uniform(.10,.58)
        points.append((rad*math.cos(theta),h,rad*math.sin(theta)))
        shades.append(rng.uniform(.08,.58)); kinds.append(3)
        glyphs.append(int(rng.choice([0,1,1,3,4,5,6])))
        seeds.append(rng.random())

    # Golden light wire traces 5.2 continuous turns on a true 3D surface.
    turns=5.15; n=1140
    f=np.linspace(0,1,n,dtype=np.float32)
    hy=.105 + f*.80
    th=TAU*(turns*f+.12)+.11*np.sin(f*TAU*3)
    rad=radius_at(hy)*(1.012+.026*np.sin(f*TAU*13))
    swag=.008*(np.sin(f*turns*TAU*1.2)**2)
    garland=np.column_stack((np.cos(th)*rad,hy-swag,np.sin(th)*rad)).astype(np.float32)

    bulb_ids=np.arange(0,n,6,dtype=np.int32)
    bulbs=garland[bulb_ids].copy()
    bulb_colors=np.array([int(rng.choice(len(COLORS),p=[.38,.29,.10,.08,.05,.10])) for _ in bulb_ids],np.int32)
    bulb_phase=rng.random(len(bulb_ids))*TAU

    ornaments=[];orn_colors=[];orn_phases=[]
    for i in range(120):
        h=rng.uniform(.05,.9)
        az=rng.uniform(0,TAU)
        rad=float(radius_at(h))*rng.uniform(.83,1.04)
        ornaments.append((math.cos(az)*rad,h,math.sin(az)*rad))
        orn_colors.append(int(rng.choice(len(COLORS),p=[.14,.13,.48,.09,.07,.09])))
        orn_phases.append(rng.uniform(0,TAU))
    orn=np.array(ornaments,np.float32)

    presents=[]
    boxes=[(-167,74,56,43,47,(180,47,59),(238,201,136),9.7),
           (-87,27,55,54,54,(39,75,167),(226,222,185),10.1),
           (94,48,60,48,56,(139,48,93),(250,199,118),10.5),
           (171,2,42,40,41,(38,125,93),(247,193,115),10.8)]
    for x,z,w,d,height,color,ribbon,at in boxes:
        pts=[];flag=[]
        for xx in np.arange(-w/2,w/2+2,4.5):
            for yy in np.arange(0,height+2,4.5):
                for zz in (-d/2,d/2):
                    pts.append((x+xx,yy/TREE_H,z+zz))
                    flag.append(abs(xx)<5 or abs(yy-height*.5)<4)
        for zz in np.arange(-d/2,d/2+2,4.5):
            for yy in np.arange(0,height+2,4.5):
                for xx in (-w/2,w/2):
                    pts.append((x+xx,yy/TREE_H,z+zz))
                    flag.append(abs(zz)<5 or abs(yy-height*.5)<4)
        for xx in np.arange(-w/2,w/2+2,4.5):
            for zz in np.arange(-d/2,d/2+2,4.5):
                pts.append((x+xx,height/TREE_H,z+zz))
                flag.append(abs(xx)<5 or abs(zz)<5)
        presents.append((np.array(pts,np.float32),np.array(flag,np.bool_),color,ribbon,at,(x,z,w,d,height)))

    pos=np.asarray(points,np.float32)
    twigs=np.asarray(segments,np.float32)
    # Each needle reacts to nearby *actual 3D bulbs*; precomputed once.
    try:
        from scipy.spatial import cKDTree
        distances,indices=cKDTree(bulbs*np.array([1,550,1])).query(pos*np.array([1,550,1]),k=1)
    except ImportError:
        indices=np.zeros(len(pos),np.int32);distances=np.full(len(pos),999,np.float32)
        b3=bulbs*np.array([1,550,1])
        for start in range(0,len(pos),1000):
            p3=pos[start:start+1000]*np.array([1,550,1])
            ds=np.sum((p3[:,None,:]-b3[None,:,:])**2,axis=2)
            indices[start:start+1000]=np.argmin(ds,axis=1)
            distances[start:start+1000]=np.sqrt(np.min(ds,axis=1))
    strengths=np.exp(-.5*(distances/32.0)**2).astype(np.float32)

    return Scene3D(pos,np.asarray(shades,np.float32),np.asarray(kinds,np.int8),
                   np.asarray(glyphs,np.uint8),np.asarray(seeds,np.float32),twigs,
                   np.asarray(indices,np.int32),strengths,
                   bulbs,bulb_colors,bulb_phase,garland,
                   orn,np.asarray(orn_colors,np.int32),np.asarray(orn_phases,np.float32),presents)


class EvergreenRenderer:
    def __init__(self, seed=2906, show_editor=True, lights=True, spin=True, snow=True):
        self.geo=build_geometry(seed)
        self.show_editor=show_editor
        self.lights_on=lights
        self.spin=spin
        self.snow_on=snow
        self.rng=np.random.default_rng(seed+5)
        self.background,self.panel=self._make_background()
        self.sky_stars=self._star_locations()
        self.flakes=self._snow_locations()
        self.source_lines=self._load_source()
        self._last_performance=None

    def _load_source(self):
        try:return Path(__file__).read_text(encoding='utf-8').splitlines()
        except Exception:return ['# EVERGREEN / VOLUMETRIC ASCII','def render(t):','    return scene(t)']

    def _make_background(self):
        yy,xx=np.mgrid[:SCENE_H,:W].astype(np.float32)
        t=yy/SCENE_H
        base=np.empty((SCENE_H,W,3),dtype=np.float32)
        base[:,:,0]=3.5+8*t
        base[:,:,1]=9.0+13*t
        base[:,:,2]=20.0+22*t
        aur=12*np.exp(-((xx-CX)/190)**2-((yy-420)/355)**2)
        base[:,:,0]+=aur*.27;base[:,:,1]+=aur*.72;base[:,:,2]+=aur*1.5
        # a faint aurora ribbon, never obscuring branches
        a=2.5*np.exp(-((xx-(CX+135*np.sin(yy/155)))/150)**2)
        base[:,:,1]+=a*.7;base[:,:,2]+=a*1.15
        img=np.uint8(clamp(base,0,255))
        forest=np.random.default_rng(828)
        for i in range(65):
            x=int((i-.5)*W/62 + forest.uniform(-9,9))
            bottom=int(forest.uniform(782,850))
            top=bottom-int(forest.uniform(20,75))
            pts=np.array([[x,top],[x-int(forest.uniform(8,20)),bottom],
                          [x+int(forest.uniform(8,20)),bottom]],np.int32)
            col=(6+int(forest.integers(0,6)),19+int(forest.integers(0,8)),26+int(forest.integers(0,12)))
            cv2.fillConvexPoly(img,pts,col)
        # A broad, gently rolling snowbank: visible from the trunk base all
        # the way to the edge of the scene, not a dark strip hidden by gifts.
        floor_points=np.array([[0,816],[77,823],[142,824],[229,811],[295,815],
                               [365,811],[477,821],[600,814],
                               [600,SCENE_H],[0,SCENE_H]],np.int32)
        cv2.fillPoly(img,[floor_points],(36,60,74))
        cv2.polylines(img,[floor_points[:8]],False,(76,104,117),1,cv2.LINE_AA)
        # Vertical blue-gray gradient with a soft pool of reflected light.
        gy,gx=np.mgrid[:SCENE_H,:W].astype(np.float32)
        snow_top=np.interp(gx, floor_points[:8,0],floor_points[:8,1])
        snow_mask=gy>=snow_top
        distance=np.maximum(0,gy-snow_top)
        pool=np.exp(-((gx-CX)/185.0)**2-((gy-837.0)/31.0)**2)
        shade=np.clip(1-distance/105,0,1)
        r=28+20*shade+22*pool
        g=40+25*shade+24*pool
        b=51+27*shade+19*pool
        img[snow_mask,0]=np.uint8(np.clip(r[snow_mask],0,255))
        img[snow_mask,1]=np.uint8(np.clip(g[snow_mask],0,255))
        img[snow_mask,2]=np.uint8(np.clip(b[snow_mask],0,255))
        cv2.polylines(img,[floor_points[:8]],False,(87,121,135),1,cv2.LINE_AA)
        # Firm contact shadows: tree and each gift are grounded in the snow.
        cv2.ellipse(img,(CX,836),(76,9),0,0,360,(15,30,35),-1,cv2.LINE_AA)
        for x,rad in [(-167,39),(-87,35),(94,37),(171,30)]:
            cv2.ellipse(img,(CX+x,843),(rad,6),0,0,360,(19,37,47),-1,cv2.LINE_AA)
        # Tiny frosty grains, short wind-carved lines, no photo textures.
        for _ in range(820):
            x=int(forest.integers(0,W))
            y=int(forest.integers(812,SCENE_H))
            if y <= snow_top[y,x]:
                continue
            val=int(forest.integers(48,107))
            img[y,x]=(min(255,val//2+10), min(255,val), min(255,val+17))
        for _ in range(55):
            x=int(forest.integers(0,W))
            y=int(forest.integers(826,SCENE_H))
            if y <= snow_top[y,x]+1:
                continue
            col=(int(forest.integers(48,74)),int(forest.integers(75,114)),int(forest.integers(96,130)))
            cv2.line(img,(x,y),(min(W-1,x+int(forest.integers(2,6))),y),col,1,cv2.LINE_AA)
        panel=np.zeros((H-SCENE_H,W,3),np.uint8)
        panel[:]=(10,13,23)
        return img,panel

    def _star_locations(self):
        rng=np.random.default_rng(142)
        return np.column_stack((rng.uniform(0,W,440),rng.uniform(0,SCENE_H-45,440),
                                rng.uniform(0,TAU,440),rng.uniform(.2,1.0,440))).astype(np.float32)

    def _snow_locations(self):
        rng=np.random.default_rng(999)
        return np.column_stack((rng.uniform(0,W,320),rng.uniform(0,SCENE_H,320),
                                rng.uniform(12,28,320),rng.uniform(0,TAU,320),
                                rng.uniform(.5,2,320))).astype(np.float32)

    def angle(self,t):
        return .60*t+.14 if self.spin else .14

    def project(self, xyz,t):
        a=self.angle(t);cs=math.cos(a);sn=math.sin(a)
        xx=xyz[:,0]*cs-xyz[:,2]*sn
        zz=xyz[:,0]*sn+xyz[:,2]*cs
        persp=1+zz*.00048
        sx=CX+xx*persp
        sy=FLOOR-xyz[:,1]*TREE_H+zz*.046
        return sx,sy,zz,persp

    def growth(self,t):
        return float(smooth((t-.65)/5.1)**.84)

    def _trunk(self, frame, t):
        """Tapered bark cylinder, anchored in snow and rotating with the tree."""
        reveal = float(smooth((t - 0.10) / 0.90))
        if reveal <= 0:
            return
        # The trunk starts slightly below ground and passes up through the
        # lowest boughs. Foliage drawn later naturally conceals its upper part.
        low, high = -0.047, -0.047 + reveal * 0.315
        levels = np.linspace(low, high, max(3, int(34 * reveal)), dtype=np.float32)
        sides = 26
        azimuths = np.arange(sides + 1) * TAU / sides
        face_quads = []
        for j in range(len(levels) - 1):
            y0 = float(levels[j]); y1 = float(levels[j + 1])
            def radius(y):
                return max(7.0, 15.0 - 24.0 * (y + 0.047)) * (1.0 + 0.035 * math.sin(y * 37))
            for i in range(sides):
                a0 = float(azimuths[i]); a1 = float(azimuths[i + 1])
                corners = np.array([
                    (radius(y0) * math.cos(a0), y0, radius(y0) * math.sin(a0)),
                    (radius(y0) * math.cos(a1), y0, radius(y0) * math.sin(a1)),
                    (radius(y1) * math.cos(a1), y1, radius(y1) * math.sin(a1)),
                    (radius(y1) * math.cos(a0), y1, radius(y1) * math.sin(a0)),
                ], np.float32)
                sx, sy, depth, _ = self.project(corners, t)
                brightness = float(clamp(0.33 + 0.50 * depth.mean() / 22.0 +
                                         0.11 * math.sin(a0 * 7.1 + j * 0.72), 0.10, 0.95))
                # Warm brown, with cracks appearing as darker facets.
                red = int(42 + brightness * 78)
                green = int(28 + brightness * 48)
                blue = int(19 + brightness * 29)
                face_quads.append((float(depth.mean()),
                                   np.column_stack((sx, sy)).astype(np.int32),
                                   (red, green, blue)))
        for _, poly, color in sorted(face_quads, key=lambda q: q[0]):
            cv2.fillConvexPoly(frame, poly, color, cv2.LINE_AA)

        # Long, uneven bark fissures wrap around the cylinder in 3D.
        for line in range(22):
            phase = line * 2.39996 + 0.3
            length = 0.026 + 0.07 * ((line * 17) % 9) / 8
            start_y = -0.027 + 0.012 * ((line * 7) % 13)
            end_y = min(high, start_y + length)
            if end_y <= start_y:
                continue
            ys = np.linspace(start_y, end_y, 9, dtype=np.float32)
            a = phase + 0.10 * np.sin(ys * 44 + line)
            rad = np.maximum(7.0, 15.0 - 24.0 * (ys + 0.047)) * 1.021
            xyz = np.column_stack((rad * np.cos(a), ys, rad * np.sin(a))).astype(np.float32)
            sx, sy, depth, _ = self.project(xyz, t)
            for k in range(len(ys) - 1):
                if (depth[k] + depth[k+1]) * 0.5 < 0:
                    continue
                col = (31, 22, 17) if line % 4 else (158, 105, 60)
                cv2.line(frame, (int(sx[k]), int(sy[k])),
                         (int(sx[k+1]), int(sy[k+1])), col, 1, cv2.LINE_AA)

        # Roots flare out on the ground and are partially submerged in snow.
        if reveal > 0.65:
            root_alpha = float(smooth((reveal - 0.65) / 0.3))
            for j in range(9):
                ang = 0.2 + TAU * j / 9 + (j % 3) * 0.12
                path = np.array([
                    (7 * math.cos(ang), -0.045, 7 * math.sin(ang)),
                    (18 * math.cos(ang), -0.049, 18 * math.sin(ang)),
                    (31 * math.cos(ang + .10), -0.052, 31 * math.sin(ang + .10)),
                    (46 * math.cos(ang + .16), -0.060, 46 * math.sin(ang + .16)),
                ], np.float32)
                sx, sy, zz, _ = self.project(path, t)
                for k in range(3):
                    if (zz[k] + zz[k + 1]) * 0.5 < -25:
                        continue
                    col = tuple(int(c * root_alpha) for c in (100, 68, 41))
                    cv2.line(frame, (int(sx[k]), int(sy[k])),
                             (int(sx[k+1]), int(sy[k+1])), col, max(1, 4-k), cv2.LINE_AA)

    def _ground_detail(self, frame, t):
        """Ground contact, snow edging, and tiny reflected lights."""
        # The tree keeps its original geometry; only a subtle patch covers
        # the buried portions of the roots at ground level.
        cv2.ellipse(frame, (CX, 840), (50, 5), 0, 0, 180, (65, 85, 92), 1, cv2.LINE_AA)
        for j in range(19):
            x = CX - 54 + j * 6
            y = 836 + int(2.0 * math.sin(j * 1.6))
            if j % 3:
                cv2.line(frame, (x, y), (x + 2, y), (87, 124, 135), 1)
        if self.lights_on and t > 6.0:
            f = float(smooth((t - 6.0) / 2.8))
            for j in range(18):
                x = CX + int((j - 9) * 15.8)
                y = 844 + (j % 4) * 6
                strength = (0.45 + 0.55 * math.sin(t * 1.9 + j * 2.2) ** 2) * f
                warm = j % 5 != 0
                rgb = (int(105 * strength), int(84 * strength), int(51 * strength)) if warm else \
                      (int(48 * strength), int(105 * strength), int(128 * strength))
                if 0 <= x < W and 0 <= y < SCENE_H:
                    cv2.line(frame, (x, y), (x + 2, y), rgb, 1, cv2.LINE_AA)

    def _sky(self,frame,t):
        for i,(x,y,ph,mag) in enumerate(self.sky_stars):
            tw=.5+.5*math.sin(t*1.1+float(ph))
            v=int((27+130*mag)*(0.56+0.44*tw))
            px=int(x);py=int(y)
            frame[py,px]=(min(255,int(v*.75)),min(255,int(v*.88)),min(255,v))
            if i%17==0 and tw>.87:
                cv2.line(frame,(px-2,py),(px+2,py),(v//2,int(v*.67),v),1)
                cv2.line(frame,(px,py-2),(px,py+2),(v//2,int(v*.67),v),1)

    def _twigs(self,frame,t,growth):
        # Thin woody boughs only: don't fill the canopy with triangles.
        seg=self.geo.twig_segments
        mask=np.maximum(seg[:,0,1],seg[:,1,1])<growth+.005
        draw=seg[mask]
        if not len(draw):return
        pr0=self.project(draw[:,0,:],t);pr1=self.project(draw[:,1,:],t)
        for i in range(0,len(draw),3):
            x0,y0,z0=pr0[0][i],pr0[1][i],pr0[2][i]
            x1,y1=pr1[0][i],pr1[1][i]
            if not (0<x0<W and 65<y0<FLOOR+25):continue
            col=(16,43,39) if z0<0 else (23,67,52)
            cv2.line(frame,(int(x0),int(y0)),(int(x1),int(y1)),col,1,cv2.LINE_AA)

    def _stamp_glyphs(self,frame,x,y,gs,colors):
        # No thousands of drawText calls. Vectorized 5×6 bitmap font rasterization.
        for glyph_id,glyph in enumerate(CHARSET):
            sel=gs==glyph_id
            if not np.any(sel):continue
            coord=GLYPH_MASKS[glyph]
            xx=x[sel,None]+coord[None,:,0]
            yy=y[sel,None]+coord[None,:,1]
            cc=colors[sel]
            frame[yy,xx]=cc[:,None,:]

    def _bulb_wave(self,t):
        ph=self.geo.bulb_phase
        shimmer=.72+.28*np.sin(4.3*t+ph)**2
        chase=.65+.35*np.sin(8.4*t+np.arange(len(ph))*.47)**2
        return shimmer*chase

    def _tree(self,frame,t):
        geo=self.geo
        grow=self.growth(t)
        if grow<=.002:return
        self._twigs(frame,t,grow)
        sx,sy,depth,scale=self.project(geo.pos,t)
        visible=(geo.pos[:,1] < grow + .009)&(sx>=5)&(sx<W-6)&(sy>70)&(sy<FLOOR+4)
        if not np.any(visible):return
        ids=np.flatnonzero(visible)
        # One frontmost 3D needle in each terminal cell: depth-aware z-buffer.
        gx=np.rint(sx[ids]/5.05).astype(np.int32)
        gy=np.rint(sy[ids]/6.8).astype(np.int32)
        keys=gy*160+gx
        order=np.lexsort((depth[ids],keys))
        order_keys=keys[order]
        chosen=order[np.r_[order_keys[1:]!=order_keys[:-1],True]]
        ix=ids[chosen]
        px=gx[chosen]*5+1;py=gy[chosen]*7+1
        safe=(px>=1)&(px<=W-7)&(py>=1)&(py<=SCENE_H-8)
        ix=ix[safe];px=px[safe];py=py[safe]
        dd=depth[ix]
        front=clamp(.42+dd/300,0,1)
        shade=geo.shade[ix]
        interior=geo.kind[ix]==3
        light_front=.43+.57*front
        base=np.column_stack((
            9+47*shade+15*front,
            32+97*shade+52*front,
            26+59*shade+31*front
        ))*light_front[:,None]
        base[interior]*=.68
        # Slight glacial highlights on drooping bough tips, never flat snow bands.
        frosted=geo.kind[ix]==2
        base[frosted]=base[frosted]*.76+np.array([74,110,107])*.24
        if self.lights_on and t>6.0:
            lit=float(smooth((t-6.0)/2.8))
            bulbs=geo.tree_light_idx[ix]
            color=COLORS[geo.bulb_colors[bulbs]]
            strength=geo.tree_light_strength[ix]*self._bulb_wave(t)[bulbs]*lit
            base+=color*(strength*.78)[:,None]

        # White sparkle travels with the growth frontier, then reveals real green needles.
        fringe=np.abs(geo.pos[ix,1]-grow)<(.024 if t<5.4 else .018)
        if t<6.0 and len(ix):
            rand=(np.sin(geo.seed[ix]*1521+t*13.7)**2)
            glow=fringe & (rand>.22)
            base[glow]=base[glow]*.22+np.array([226,249,238])*.78
        # tiny branch highlights, physically oriented by camera depth
        twinkle=(np.sin(t*1.65+geo.seed[ix]*24)**2)*4.8
        base+=twinkle[:,None]
        self._stamp_glyphs(frame,px,py,geo.glyph[ix],np.uint8(clamp(base,0,255)))

    def _garland(self,frame,bloom,t,front):
        if not self.lights_on:return
        reveal=float(smooth((t-6.25)/2.4))
        if reveal<=0:return
        g=self.geo
        n=int(len(g.garland)*reveal)
        if n<2:return
        sx,sy,zz,_=self.project(g.garland[:n],t)
        points=np.column_stack((sx,sy)).astype(np.int32)
        for i in range(1,n,3):
            is_front=zz[i]>=-3
            if is_front!=front:continue
            p1=points[i-1];p2=points[i]
            if p1[1]<70 or p1[1]>840:continue
            cv2.line(frame,tuple(p1),tuple(p2),(60,104,139) if front else (27,58,65),1,cv2.LINE_AA)
        # Lights are geometrically attached to the 3D spiral, NOT a 2D overlay.
        bulb_count=max(0,int(len(g.bulbs)*reveal))
        if not bulb_count:return
        x,y,z,_=self.project(g.bulbs[:bulb_count],t)
        intensities=self._bulb_wave(t)
        for i in range(bulb_count):
            if (z[i]>-4)!=front:continue
            xx=int(x[i]);yy=int(y[i]);depth=z[i]
            if xx<5 or xx>=W-5 or yy<75 or yy>838:continue
            c=COLORS[g.bulb_colors[i]]
            power=float(float(intensities[i])*(.58+.42*float(clamp(.5+depth/250))))
            radius=1 if not front else 2+(i%7==0)
            col=tuple(int(q) for q in c*power)
            cv2.circle(frame,(xx,yy),radius,col,-1,cv2.LINE_AA)
            cv2.circle(bloom,(xx,yy),5 if front else 3,tuple(float(q)*power for q in c),-1,cv2.LINE_AA)
            if front and i%6==0 and power>.72:
                cv2.line(frame,(xx-4,yy),(xx+4,yy),col,1,cv2.LINE_AA)
                cv2.line(frame,(xx,yy-4),(xx,yy+4),col,1,cv2.LINE_AA)

    def _ornaments(self,frame,bloom,t):
        if t<7.45:return
        g=self.geo
        x,y,z,_=self.project(g.ornaments,t)
        for i in range(len(g.ornaments)):
            appear=float(smooth((t-7.45-i*.025)/.78))
            if appear<=.05:continue
            xx=int(x[i]);yy=int(y[i]);depth=z[i]
            if not (10<xx<W-10 and 85<yy<842):continue
            facing=float(clamp(.42+depth/250))
            if facing<.22:continue
            color=COLORS[g.ornament_colors[i]]
            shine=.76+.24*math.sin(t*2.5+float(g.ornament_phase[i]))**2
            rad=max(1,round((2.2+1.6*facing)*appear))
            c=tuple(int(v*shine*(.55+.45*facing)) for v in color)
            cv2.line(frame,(xx,yy-6),(xx,yy-rad), (79,105,99),1,cv2.LINE_AA)
            cv2.circle(frame,(xx,yy),rad,(int(c[0]*.65),int(c[1]*.65),int(c[2]*.65)),-1,cv2.LINE_AA)
            cv2.circle(frame,(xx-rad//2,yy-rad//2),1,(230,246,250),-1,cv2.LINE_AA)
            if facing>.55 and i%2==0:
                cv2.circle(bloom,(xx,yy),rad+3,tuple(float(v)*.62*appear for v in color),-1,cv2.LINE_AA)

    def _gifts(self,frame,t):
        g=self.geo
        # Gifts stay on the ground while the camera orbits the tree.
        for points,flags,base,ribbon,start,box in g.gifts:
            progress=float(smooth((t-start)/.9))
            if progress<=0:continue
            pp=points.copy();pp[:,1]+=(1-progress)*.053-35/TREE_H
            px,py,depth,_=self.project(pp,t)
            order=np.argsort(depth)
            for i in order[::2]:
                x=int(px[i]);y=int(py[i]);
                if 0<x<W-3 and 40<y<SCENE_H-4:
                    color=ribbon if flags[i] else base
                    face=float(clamp(.60+depth[i]/440,.42,1))
                    rgb=tuple(int(c*face) for c in color)
                    cv2.putText(frame,'+' if flags[i] else '#',(x,y),cv2.FONT_HERSHEY_PLAIN,.45,rgb,1,cv2.LINE_AA)
            # Woven gold bow above the gift, also orbiting in 3D
            x,z,w,d,height=box
            bow=[]
            for j in range(18):
                f=j/17
                ang=f*TAU*1.7
                bow.append((x+math.sin(ang)*10,(height-35)/TREE_H+(7+7*math.sin(ang))/TREE_H,z+math.cos(ang)*6))
            bx,by,_,_=self.project(np.array(bow,np.float32),t)
            for j in range(1,len(bx)):
                cv2.line(frame,(int(bx[j-1]),int(by[j-1])),(int(bx[j]),int(by[j])),ribbon,1,cv2.LINE_AA)

    def _star(self,frame,bloom,t):
        strength=float(smooth((t-10.65)/1.8))
        if strength<=0:return
        # Articulated 8-point jewel with a physically spinning face.
        x=CX;y=TOP-17 + math.sin(t*1.35)*1.0
        pulse=(.87+.13*math.sin(t*3.8)**2)
        spin=t*.44
        size=(10+13*strength)*pulse
        points=stars_8(x,y,size,spin)
        cv2.fillPoly(frame,[points],(255,226,119),cv2.LINE_AA)
        cv2.polylines(frame,[points],True,(255,247,194),1,cv2.LINE_AA)
        cv2.circle(frame,(int(x),int(y)),max(2,int(3*strength)),(255,255,245),-1,cv2.LINE_AA)
        # Chromatic light shafts (different lengths, asymmetric twinkle).
        for i in range(8):
            a=i*TAU/8-spin*.25
            length=(18+27*strength)*(1.18 if i%2==0 else .66)*pulse
            x1=int(x+math.cos(a)*length);y1=int(y+math.sin(a)*length)
            x2=int(x-math.cos(a)*length);y2=int(y-math.sin(a)*length)
            cv2.line(frame,(x1,y1),(x2,y2),(218,192,111),1,cv2.LINE_AA)
        cv2.circle(bloom,(int(x),int(y)),int(8+strength*13),(255*strength,225*strength,145*strength),-1,cv2.LINE_AA)
        for k in range(10):
            phi=t*1.45+k*TAU/10
            rr=31+8*math.sin(t*1.9+k)
            xx=int(x+math.cos(phi)*rr);yy=int(y+math.sin(phi)*rr)
            cv2.circle(frame,(xx,yy),1,(255,236,170),-1,cv2.LINE_AA)

    def _snow(self,frame,t):
        if not self.snow_on:return
        for i,(x,y,spd,ph,sz) in enumerate(self.flakes):
            xx=int((x+math.sin(t*.52+float(ph))*10+t*1.8)%W)
            yy=int((y+t*spd)%(SCENE_H+17))
            if yy>=SCENE_H:continue
            l=int(105+70*(.5+.5*math.sin(t*.6+float(ph))**2))
            if sz>1.6 and i%5==0:
                cv2.line(frame,(xx-1,yy),(xx+1,yy),(l,l+10,min(l+25,255)),1)
                cv2.line(frame,(xx,yy-1),(xx,yy+1),(l,l+10,min(l+25,255)),1)
            else:
                frame[yy,xx]=(l,l+7,min(l+20,255))

    def _post_glow(self,frame,bloom):
        # Two Gaussian scales: sharp warm bulb bloom + diffused light spill.
        blur_small=cv2.GaussianBlur(bloom,(0,0),4)
        blur_large=cv2.GaussianBlur(bloom,(0,0),16)
        lighting=blur_small*1.09+blur_large*.80
        return np.uint8(clamp(frame.astype(np.float32)+lighting,0,255))

    def _message(self,frame,t):
        if t<12.8:return
        message='M E R R Y   C H R I S T M A S'
        amount=min(len(message),int((t-12.8)*11.6))
        if amount<=0:return
        size=.51
        (tw,_),_=cv2.getTextSize(message,cv2.FONT_HERSHEY_SIMPLEX,size,1)
        xx=(W-tw)//2;yy=860
        cv2.putText(frame,message[:amount],(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,size,(210,184,146),1,cv2.LINE_AA)

    def _telemetry(self,frame,t):
        text1=f'rotation {self.angle(t)*180/math.pi%360:06.2f}deg'
        text2=f'3D needles {len(self.geo.pos):,}  |  bulbs {len(self.geo.bulbs)}'
        cv2.putText(frame,text1,(12,19),cv2.FONT_HERSHEY_PLAIN,.69,(117,145,164),1,cv2.LINE_AA)
        cv2.putText(frame,text2,(12,33),cv2.FONT_HERSHEY_PLAIN,.69,(117,145,164),1,cv2.LINE_AA)
        cv2.putText(frame,f'{t:05.2f}s',(W-67,14),cv2.FONT_HERSHEY_PLAIN,.75,(191,187,151),1,cv2.LINE_AA)

    def _editor(self,t):
        panel=self.panel.copy()
        cv2.rectangle(panel,(0,0),(W-1,29),(21,27,40),-1)
        cv2.line(panel,(0,0),(W,0),(38,53,72),1)
        cv2.putText(panel,'evergreen.py',(10,20),cv2.FONT_HERSHEY_PLAIN,.86,(186,204,217),1,cv2.LINE_AA)
        cv2.circle(panel,(W-130,16),3,(249,206,106),-1,cv2.LINE_AA)
        cv2.putText(panel,'running | 24 fps',(W-122,20),cv2.FONT_HERSHEY_PLAIN,.7,(145,183,185),1,cv2.LINE_AA)
        total=len(self.source_lines)
        top=int(t*19) % max(1,total-13)
        for i in range(13):
            lineno=top+i
            y=42+i*10
            if lineno>=total:break
            source=self.source_lines[lineno].expandtabs(3).rstrip()[:94]
            if i==6:cv2.rectangle(panel,(0,y-9),(W,y+1),(18,24,42),-1)
            cv2.putText(panel,str(lineno+1).rjust(3),(5,y),cv2.FONT_HERSHEY_PLAIN,.64,(78,95,113),1,cv2.LINE_AA)
            col=(141,163,180)
            if source.lstrip().startswith('#'):col=(101,148,133)
            elif source.lstrip().startswith(('def ','class ')):col=(156,165,232)
            elif '"""' in source:col=(189,186,146)
            elif source.lstrip().startswith(('for ','if ','return ','while ')):col=(184,141,192)
            cv2.putText(panel,source,(39,y),cv2.FONT_HERSHEY_PLAIN,.68,col,1,cv2.LINE_AA)
        return panel

    def render(self,t):
        frame=self.background.copy()
        bloom=np.zeros_like(frame,np.float32)
        self._sky(frame,t)
        # The wooden trunk rotates with the tree. Foliage hides its upper part.
        self._trunk(frame,t)
        # rear lights appear behind textured glyphs; front lights go on top
        self._garland(frame,bloom,t,front=False)
        self._tree(frame,t)
        self._garland(frame,bloom,t,front=True)
        self._ornaments(frame,bloom,t)
        self._ground_detail(frame,t)
        self._gifts(frame,t)
        self._star(frame,bloom,t)
        frame=self._post_glow(frame,bloom)
        self._snow(frame,t)
        self._message(frame,t)
        self._telemetry(frame,t)
        if self.show_editor:
            panel=self._editor(t)
        else:
            panel=np.repeat(frame[-1:, :, :],H-SCENE_H,axis=0)
        return np.vstack([frame,panel])


def export_video(renderer, output, duration=20, fps=24, scale=1.0):
    """Export using short, reliable OpenCV chunks and optional FFmpeg finishing.

    The renderer never allocates hundreds of PNGs. Each chunk is a proper MP4,
    keeping large projects manageable on an ordinary laptop. FFmpeg is only
    required to combine and optionally upscale those chunks.
    """
    total = max(1, int(round(fps * duration)))
    width = max(2, round(W * scale / 2) * 2)
    height = max(2, round(H * scale / 2) * 2)
    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        # Useful fallback for systems that have only the pip dependencies.
        if scale != 1:
            print('FFmpeg not installed: exporting at native resolution.', flush=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
        if not writer.isOpened():
            raise RuntimeError('Could not open MP4 writer.')
        try:
            for i in range(total):
                writer.write(cv2.cvtColor(renderer.render(i/fps), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print('Saved', output, flush=True)
        return

    with tempfile.TemporaryDirectory(prefix='evergreen_render_') as tmp:
        segfiles=[]
        chunk = max(1, int(fps * 5.0))
        for seg_idx,start in enumerate(range(0,total,chunk)):
            end=min(start+chunk,total)
            path=Path(tmp)/f'chunk_{seg_idx:03d}.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
            if not writer.isOpened():
                raise RuntimeError(f'Unable to open {path}')
            try:
                for i in range(start,end):
                    writer.write(cv2.cvtColor(renderer.render(i/fps),cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            segfiles.append(path)
            print(f'Frames rendered: {end}/{total}', flush=True)
        index=Path(tmp)/'segments.txt'
        index.write_text(''.join(f"file '{p.as_posix()}'\n" for p in segfiles),encoding='utf-8')
        command=['ffmpeg','-y','-hide_banner','-loglevel','error',
                 '-f','concat','-safe','0','-i',str(index),
                 '-vf',f'scale={width}:{height}:flags=lanczos',
                 '-c:v','libx264','-preset','veryfast','-crf','25',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        subprocess.run(command,check=True)
        print(f'Video ready: {output}', flush=True)


def live(renderer):
    """Real-time desktop animation. Tkinter ships with standard Windows Python."""
    import tkinter as tk
    from PIL import ImageTk
    root=tk.Tk()
    root.title('EVERGREEN / VOLUMETRIC 3D ASCII TREE')
    root.configure(bg='#040915')
    label=tk.Label(root,bd=0,highlightthickness=0,bg='#040915')
    label.pack()
    clock=time.perf_counter()
    state={'started':clock,'paused':False,'pause_time':0.,'full':False,'current_img':None,'running':True}

    def key(e):
        k=e.keysym.lower()
        if k in ('escape','q'):
            state['running']=False;root.destroy();return
        if k=='r':state['started']=time.perf_counter();state['paused']=False
        if k=='space':
            if state['paused']:
                state['started']=time.perf_counter()-state['pause_time']
            else:state['pause_time']=time.perf_counter()-state['started']
            state['paused']=not state['paused']
        if k=='s':renderer.spin=not renderer.spin
        if k=='l':renderer.lights_on=not renderer.lights_on
        if k=='n':renderer.snow_on=not renderer.snow_on
        if k=='e':renderer.show_editor=not renderer.show_editor
        if k=='f':state['full']=not state['full'];root.attributes('-fullscreen',state['full'])

    def tick():
        if not state['running']:return
        started=time.perf_counter()
        t=state['pause_time'] if state['paused'] else (started-state['started'])%20.0
        frame=renderer.render(t)
        # Scale uniformly, retaining pixel ratio on all displays.
        sw=root.winfo_screenwidth();sh=root.winfo_screenheight()-55
        scale=min(1.0,sw/W,sh/H)
        im=Image.fromarray(frame)
        if scale<1: im=im.resize((max(1,int(W*scale)),max(1,int(H*scale))),Image.Resampling.LANCZOS)
        state['current_img']=ImageTk.PhotoImage(im)
        label.config(image=state['current_img'])
        delay=max(1,int(1000/FPS-(time.perf_counter()-started)*1000))
        root.after(delay,tick)

    root.bind('<Key>',key)
    root.protocol('WM_DELETE_WINDOW',lambda:(state.update(running=False),root.destroy()))
    tick();root.mainloop()


def main():
    cli=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--export',metavar='OUTPUT.MP4')
    cli.add_argument('--still',type=float,metavar='SECOND')
    cli.add_argument('--seconds',type=float,default=20)
    cli.add_argument('--fps',type=int,default=FPS)
    cli.add_argument('--scale',type=float,default=1.0,help='Export resolution multiplier, e.g. 1.8 for ~1080p portrait')
    cli.add_argument('--no-editor',action='store_true')
    cli.add_argument('--no-snow',action='store_true')
    cli.add_argument('--no-lights',action='store_true')
    args=cli.parse_args()
    r=EvergreenRenderer(show_editor=not args.no_editor,
                         lights=not args.no_lights,snow=not args.no_snow)
    if args.still is not None:
        out=Path('evergreen_preview.png')
        Image.fromarray(r.render(args.still)).save(out)
        print('Saved',out.resolve(),flush=True)
    elif args.export:
        export_video(r,args.export,args.seconds,args.fps,args.scale)
    else:
        live(r)


if __name__=='__main__':main()
