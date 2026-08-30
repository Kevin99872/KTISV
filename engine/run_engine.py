"""PyInstaller 進入點。

不能直接把 ktisv_engine/__main__.py 交給 PyInstaller —— 它會被當成頂層
`__main__` 執行,套件內的相對匯入(from . import ...)就會失效。
這個檔案以正常方式匯入套件,再呼叫其 main。
"""

import multiprocessing

from ktisv_engine.__main__ import main

if __name__ == "__main__":
    # 一定要在任何其他事情之前呼叫。
    #
    # 打包後的 exe 沒有「python.exe + 腳本」這種結構。multiprocessing 用
    # spawn 建子行程時,子行程會**從頭重新執行整個 exe** —— 於是又起一個
    # 引擎、又印一次握手、又去搶一個埠,前端拿到的資訊全亂掉。
    #
    # freeze_support() 就是讓子行程在這一行認出自己的身分,去跑被指派的
    # 工作而不是重跑 main。
    #
    # 以前沒這行也相安無事,因為引擎的相依裡沒有人用多行程。把 torch /
    # demucs 內建進來之後就不成立了。這種錯誤在原始碼樹**永遠重現不出來**
    # (那時真的有 python.exe),只會在打包版發作。
    multiprocessing.freeze_support()
    raise SystemExit(main())
