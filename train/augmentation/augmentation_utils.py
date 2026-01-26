#!/usr/bin/env python

import os
    


def parse_yolo_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            labels.append([int(cls), float(xc), float(yc), float(w), float(h)])
    return labels


def save_yolo_labels(labels, output_path):
    with open(output_path, "w") as f:
        for cls, xc, yc, w, h in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
