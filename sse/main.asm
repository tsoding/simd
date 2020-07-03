%define SYS_EXIT 60

segment .data
src:   db 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15, 1, 15
shift: db 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13
cap:   db 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25
cap0:   db 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26


segment .text
global _start

rot13_16b:
    movaps xmm0, [rax]
    movaps xmm1, [shift]
    paddb xmm0, xmm1
    movaps xmm2, xmm0

    movaps xmm1, [cap]
    pcmpgtb xmm0, xmm1

    movaps xmm1, [cap0]
    pand xmm0, xmm1

    psubb xmm2, xmm0
    movaps [rax], xmm2
    ret

_start:
    mov rax, src
    call rot13_16b
    call rot13_16b

    mov rax, SYS_EXIT
    mov rdi, 0
    syscall
    ret
