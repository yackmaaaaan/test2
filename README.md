# HaseLab 2段階 写真認識テスト v4

YOLO26 segmentation raw output [1,37,8400] を Ultralytics 仕様に合わせて解析。
channel 4 は sigmoid を再適用せず tile class score として使用。
全景写真向けに牌らしい縦横比と横一列クラスタを選択する後処理を追加。
