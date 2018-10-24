#! /usr/bin/env python3
# -*- coding: utf-8 -*-

vertex = """
    attribute vec2 position;
    attribute vec2 textcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
"""

fragment = """
    uniform sampler2D texture;
    varying vec2 texcoord;
    void main()
    {
        gl_Fragcolor = texture2D(texture, v_texcoord);
    }
"""
