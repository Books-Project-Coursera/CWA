"""
Strategy 2 - Object Detection package (Ultralytics YOLO + Pascal VOC).

Kế thừa convention của nhánh Strategy2_TinyImageNet (config-driven, CLI
override, export Excel bằng pandas/openpyxl) nhưng toàn bộ train/val/export
model đều giao cho Ultralytics API — không tự viết training loop.

Entrypoint: main_detection.py (subcommand train / eval / export / export-model)
"""
