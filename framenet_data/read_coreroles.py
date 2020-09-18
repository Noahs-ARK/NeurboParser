#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:12:34 2017

@author: hpeng
"""
import xml.etree.ElementTree as et
import sys, os, io

FRAME_DIR = 'fndata-1.5/frame'
def read_fes_lus(frame_file):
    f = open(frame_file, "rb")
    #    with codecs.open(luIndex_file, "r", "utf-8") as xml_file: # TODO: why won't this right way of reading work?
    tree = et.parse(f)
    root = tree.getroot()

    frcount, fes, core_fes = 0, [], []
    for frame in root.iter('{http://framenet.icsi.berkeley.edu}frame'):
        framename = frame.attrib["name"]
        frcount += 1

    if frcount > 1:
        raise Exception("More than one frame?", frame_file, framename)

    n_role, n_core = 0, 0
    for fe in root.iter('{http://framenet.icsi.berkeley.edu}FE'):
        fename = fe.attrib["name"]
        fes.append(fename)
        n_role += 1
        if fe.attrib["coreType"] == "Core":
            core_fes.append(fename)
            n_core += 1
    print n_core, '/', n_role


    return framename, fes, core_fes

def read_frame_maps():
    sys.stderr.write("reading the frame-element - frame map from " + FRAME_DIR + "...\n")
    
    fout = io.open('frames', 'w', encoding = 'utf-8')

    for f in os.listdir(FRAME_DIR):
        framef = os.path.join(FRAME_DIR, f)
        if framef.endswith("xsl"):
            continue
        frame_name, fes, core_fes = read_fes_lus(framef)
        fout.write(frame_name + u'\n')
        fout.write(u'\t'.join(fes) + u'\n')
        fout.write(u'\t'.join(core_fes) + u'\n')
    fout.close()
        
        

    
if __name__ == '__main__':
    read_frame_maps()
