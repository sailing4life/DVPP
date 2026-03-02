#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parser for ORC DXT certificate files (XML format).

Extracts sail geometry (dimensions) for use by the sail force model.
All lengths are in metres as stored in the DXT (Metric=1 assumed).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class MainsailGeom:
    """Mainsail rig geometry extracted from an ORC DXT file."""
    P: float       # luff length (m)
    E: float       # foot length (m)
    BAS: float     # boom above sheerline (m)
    HB: float      # boom girth height (m)
    MGT: float     # girth at top-quarter station
    MGU: float     # girth at upper-mid station
    MGM: float     # girth at mid station
    MGL: float     # girth at lower-quarter station


@dataclass
class HeadsailGeom:
    """Jib or flying-headsail geometry extracted from an ORC DXT file."""
    sail_id: str
    is_flying: bool    # True = flying (code zero / A-sail), False = non-flying jib
    JH: float          # haul height / boom height (m)
    JGT: float         # girth at top-quarter station
    JGU: float         # girth at upper-mid station
    JGM: float         # girth at mid station (hhw)
    JGL: float         # girth at lower-mid station
    LPG: float         # LP measurement / foot depth (m)
    JIBLUFF: float     # luff length (m)
    sail_area: float   # measured sail area (m²)
    comment: str = ""

    def label(self):
        kind = "flying" if self.is_flying else "jib"
        parts = [self.sail_id, f"{self.sail_area:.0f} m²", kind]
        if self.comment:
            parts.append(self.comment)
        return "  ·  ".join(parts)


@dataclass
class AsymSpinGeom:
    """Asymmetric spinnaker geometry extracted from an ORC DXT asym_spin record."""
    sail_id: str
    SLU: float      # spinnaker luff (m)
    SLE: float      # spinnaker leech (m)
    ASL: float      # asymmetric half-width / mid length (m)
    AMG: float      # maximum girth (m)
    ASF: float      # foot girth / spread (m)
    sail_area: float
    comment: str = ""

    def label(self):
        parts = [self.sail_id, f"{self.sail_area:.0f} m²", "asym spin"]
        if self.comment:
            parts.append(self.comment)
        return "  ·  ".join(parts)


@dataclass
class ORCRig:
    """Parsed ORC DXT certificate — rig dimensions and sail inventory."""
    yacht_name: str
    cert_no: str
    P: float
    E: float
    BAS: float
    IG: float
    ISP: float
    J: float
    mainsail: MainsailGeom
    headsails: list = field(default_factory=list)
    asym_spinners: list = field(default_factory=list)

    def jibs(self):
        """Non-flying jib headsails."""
        return [h for h in self.headsails if not h.is_flying]

    def flying_headsails(self):
        """Flying headsails (code zeros, A-sails) and asymmetric spinnakers combined."""
        return [h for h in self.headsails if h.is_flying] + self.asym_spinners


def _fval(elem, fieldname, default=0.0):
    el = elem.find(f"FIELD[@fieldname='{fieldname}']")
    if el is None:
        return default
    v = el.get("value", "")
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def _sval(elem, fieldname, default=""):
    el = elem.find(f"FIELD[@fieldname='{fieldname}']")
    if el is None:
        return default
    return el.get("value", default) or default


def parse_dxt(source):
    """
    Parse an ORC DXT certificate file.

    Parameters
    ----------
    source : str or file-like
        Path to the .dxt file, or any file-like object (e.g. Streamlit UploadedFile).

    Returns
    -------
    ORCRig
    """
    if hasattr(source, "read"):
        tree = ET.parse(source)
    else:
        tree = ET.parse(source)
    root = tree.getroot()
    inp = root.find("INPUT")
    if inp is None:
        raise ValueError("No <INPUT> element found in DXT file.")

    def fv(name, default=0.0):
        el = inp.find(f".//FIELD[@fieldname='{name}']")
        if el is None:
            return default
        v = el.get("value", "")
        try:
            return float(v) if v else default
        except (ValueError, TypeError):
            return default

    def sv(name, default=""):
        el = inp.find(f".//FIELD[@fieldname='{name}']")
        if el is None:
            return default
        return el.get("value", default) or default

    P   = fv("P")
    E   = fv("E")
    BAS = fv("BAS")
    IG  = fv("IG")
    ISP = fv("ISP")
    J   = fv("J")

    # ── Mainsail ─────────────────────────────────────────────────────────────
    main_records = inp.findall(".//SAIL[@SailCode='main']/RECORD")
    if main_records:
        r = main_records[0]
        mainsail = MainsailGeom(
            P=P, E=E, BAS=BAS,
            HB=_fval(r, "HB"),
            MGT=_fval(r, "MGT"),
            MGU=_fval(r, "MGU"),
            MGM=_fval(r, "MGM"),
            MGL=_fval(r, "MGL"),
        )
    else:
        mainsail = MainsailGeom(P=P, E=E, BAS=BAS, HB=0.0, MGT=0.0, MGU=0.0, MGM=0.0, MGL=0.0)

    # ── Headsails (jib SailCode contains non-flying and flying) ──────────────
    headsails = []
    for r in inp.findall(".//SAIL[@SailCode='jib']/RECORD"):
        headsails.append(HeadsailGeom(
            sail_id=_sval(r, "SailId"),
            is_flying=bool(int(_fval(r, "Flying", 0))),
            JH=_fval(r, "JH"),
            JGT=_fval(r, "JGT"),
            JGU=_fval(r, "JGU"),
            JGM=_fval(r, "JGM"),
            JGL=_fval(r, "JGL"),
            LPG=_fval(r, "LPG"),
            JIBLUFF=_fval(r, "JIBLUFF"),
            sail_area=_fval(r, "SailArea"),
            comment=_sval(r, "Comment"),
        ))

    # ── Asymmetric spinnakers ─────────────────────────────────────────────────
    asym_spinners = []
    for r in inp.findall(".//SAIL[@SailCode='asym_spin']/RECORD"):
        asym_spinners.append(AsymSpinGeom(
            sail_id=_sval(r, "SailId"),
            SLU=_fval(r, "SLU"),
            SLE=_fval(r, "SLE"),
            ASL=_fval(r, "ASL"),
            AMG=_fval(r, "AMG"),
            ASF=_fval(r, "ASF"),
            sail_area=_fval(r, "SailArea"),
            comment=_sval(r, "Comment"),
        ))

    return ORCRig(
        yacht_name=sv("YachtName"),
        cert_no=sv("CertNo"),
        P=P, E=E, BAS=BAS, IG=IG, ISP=ISP, J=J,
        mainsail=mainsail,
        headsails=headsails,
        asym_spinners=asym_spinners,
    )
