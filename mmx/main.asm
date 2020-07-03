%define SYS_EXIT 60

segment .data
src:   db 15, 1, 15, 1, 15, 1, 15, 1
shift: db 13, 13, 13, 13, 13, 13, 13, 13
cap:   db 25, 25, 25, 25, 25, 25, 25, 25
cap0:  db 26, 26, 26, 26, 26, 26, 26, 26

segment .text
global _start

rot13_8b:
    movq mm0, rax
    movq mm1, [shift]
    paddb mm0, mm1
    movq mm2, mm0
    movq rax, mm0

    movq mm1, [cap]
    pcmpgtb mm0, mm1
    movq rax, mm0

    movq mm1, [cap0]
    pand mm0, mm1
    movq rax, mm0

    psubb mm2, mm0
    movq rax, mm2
    ret

_start:
    mov rax, [src]
    call rot13
    call rot13

    mov rax, SYS_EXIT
    mov rdi, 0
    syscall
    ret
