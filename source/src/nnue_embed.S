.section .rodata
.balign 64
.global gEmbeddedNNUEData
.global gEmbeddedNNUEEnd
gEmbeddedNNUEData:
.incbin "source/data/current.nnue"
gEmbeddedNNUEEnd:
