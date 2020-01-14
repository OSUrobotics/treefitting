#!/usr/bin/env python3

def pca_ratios_cylinder():
    """
    Trim ratios for pca
       For radius 0.01, 0.09  height 0.36
       pca ratio min 2.50 max 34.41, avg 9.99
       For radius 0.01, 0.09  height 0.06
       pca ratio min 1.65 max 8.09, avg 5.09
         Cylinder has a method to find these given radius range and height
    :return: struct containing ratios (labeled)
    """
    pca_ratios = {}
    pca_ratios["pca_ratio"] = 7.0
    pca_ratios["pca_min"] = 1.6
    pca_ratios["pca_max"] = 40.0
    return pca_ratios

def radius_and_height():
    return {"radius_min":0.015, "radius_max":0.09, "height":4*0.09}

def best_pts():
    best_pts = {}
    best_pts[245401] = "Thin"
    best_pts[44221] = "Thin"
    best_pts[228185] = "Thin"
    best_pts[37763] = "Thin connect"
    best_pts[49041] = "Trunk"
    best_pts[220033] = "Thin"
    best_pts[206383] = "Trunk"
    return best_pts


def bad_pts():
    bad_pts = {}
    bad_pts[259144] = "Noisy end"
    bad_pts[244173] = "Bushy"
    bad_pts[130202] = "Bushy"
    bad_pts[156572] = "Bushy"
    return bad_pts
