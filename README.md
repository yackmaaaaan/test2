# 麻雀牌 写真認識テスト HaseLab 3段階版 v5

認識パイプライン:
1. regions-yolo-26n.onnx で self_hand（自分の手牌領域）を検出
2. その領域だけを tile-yolo-26n.onnx に入力して各牌を検出
3. 各牌を224×224へ整形して classifier-resnet50.onnx で39クラス分類

GitHub Pagesではフォルダ構成を維持して配置してください。`models/regions-yolo-26n.onnx` を含みます。
