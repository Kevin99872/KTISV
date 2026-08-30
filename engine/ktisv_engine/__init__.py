"""KTISV 音訊引擎。

即時音訊路由 / 人聲分離 / 雙路輸出後端。由 Avalonia 前端透過
localhost TCP (JSON-lines) 驅動,也可以單獨當函式庫使用。
"""

__version__ = "0.1.0"

SAMPLE_RATE = 48000

# 2.7 ms @ 48 kHz。480 曾是預設值,但實機量測顯示 block 大小幾乎全額
# 反映在耳返延遲上(獨佔模式:480 → 31 ms、128 → 13 ms),而混音本身
# 只吃掉 128 取樣預算的約 5%,沒有理由留那麼大的緩衝。
BLOCK_SIZE = 128
