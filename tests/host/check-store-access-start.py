#!/usr/bin/env python3
"""A store that faults across a page boundary is emulated at its first byte.

When an unaligned store starts on a host-writable page and runs into a
non-writable one, the CPU reports the first byte of the second page as the
fault address, not the store's first byte, and writes nothing. The alias store
emulator wrote single-register stores at the fault address, so such a store
landed at the page boundary. This runs real stores of every addressing mode the
emulator decodes across a 16KB boundary, with the first page writable and with
both pages protected, catches the fault, and checks that the production
ios_store_access_start() recovers the store's real first byte from the faulting
instruction and registers. Loads, LSE atomics and pairs must be left alone.
Needs an arm64 Mac (the CPU behaviour is the point); uses ASan/UBSan.
"""
from pathlib import Path
import os
import platform
import subprocess
import sys
import tempfile

if platform.system() != 'Darwin' or platform.machine() != 'arm64':
    print('SKIP: needs an arm64 Mac')
    sys.exit(0)

root = Path(__file__).resolve().parents[2]
source = (root / 'build/ntdll-unix/signal_arm64_ios.c').read_text()

def function(signature):
    start = source.index(signature)
    return source[start:source.index('\n}', start) + 2] + '\n'

assert 'start = ios_store_access_start( insn, gpr, fault_addr );' in source

code = r'''
#include <assert.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
'''
code += function('static uintptr_t ios_store_access_start( uint32_t insn, const uint64_t gpr[32], uintptr_t fault_addr )')
code += r'''
static volatile uintptr_t seen_fault, seen_start;
static volatile int faults;

static void on_fault(int sig, siginfo_t *info, void *context)
{
    ucontext_t *uc = context;
    uint64_t gpr[32];
    uint32_t insn;
    (void)sig;
    memcpy(&insn, (const void *)uc->uc_mcontext->__ss.__pc, sizeof(insn));
    memcpy(gpr, uc->uc_mcontext->__ss.__x, sizeof(uc->uc_mcontext->__ss.__x));
    gpr[29] = uc->uc_mcontext->__ss.__fp;
    gpr[30] = uc->uc_mcontext->__ss.__lr;
    gpr[31] = uc->uc_mcontext->__ss.__sp;
    seen_fault = (uintptr_t)info->si_addr;
    seen_start = ios_store_access_start(insn, gpr, seen_fault);
    faults++;
    uc->uc_mcontext->__ss.__pc += 4;   /* skip the store; nothing was written */
}

/* Each case stores `width` bytes whose first byte is `target`. */
static void str_x_uimm(uint8_t *target) { uint64_t v = 0x1122334455667788ull; uint8_t *b = target - 16; __asm__ volatile("str %0, [%1, #16]" :: "r"(v), "r"(b) : "memory"); }
static void str_w_uimm(uint8_t *target) { uint32_t v = 0x11223344u; uint8_t *b = target - 8; __asm__ volatile("str %w0, [%1, #8]" :: "r"(v), "r"(b) : "memory"); }
static void strh_uimm(uint8_t *target) { uint32_t v = 0x1122u; uint8_t *b = target - 2; __asm__ volatile("strh %w0, [%1, #2]" :: "r"(v), "r"(b) : "memory"); }
static void stur_x(uint8_t *target) { uint64_t v = 0x1122334455667788ull; uint8_t *b = target + 5; __asm__ volatile("stur %0, [%1, #-5]" :: "r"(v), "r"(b) : "memory"); }
static void str_x_post(uint8_t *target) { uint64_t v = 0x1122334455667788ull; uint8_t *b = target; __asm__ volatile("str %1, [%0], #16" : "+r"(b) : "r"(v) : "memory"); }
static void str_x_pre(uint8_t *target) { uint64_t v = 0x1122334455667788ull; uint8_t *b = target + 8; __asm__ volatile("str %1, [%0, #-8]!" : "+r"(b) : "r"(v) : "memory"); }
static void str_x_reg_lsl(uint8_t *target) { uint64_t v = 0x1122334455667788ull, i = 5; uint8_t *b = target - 5 * 8; __asm__ volatile("str %0, [%1, %2, lsl #3]" :: "r"(v), "r"(b), "r"(i) : "memory"); }
static void str_w_reg_sxtw(uint8_t *target) { uint32_t v = 0x11223344u; int32_t i = -3; uint8_t *b = target + 3 * 4; __asm__ volatile("str %w0, [%1, %w2, sxtw #2]" :: "r"(v), "r"(b), "r"(i) : "memory"); }
static void str_x_reg_uxtw(uint8_t *target) { uint64_t v = 0x1122334455667788ull; uint32_t i = 24; uint8_t *b = target - 24; __asm__ volatile("str %0, [%1, %w2, uxtw]" :: "r"(v), "r"(b), "r"(i) : "memory"); }
static void str_q_uimm(uint8_t *target) { uint8_t *b = target - 32; __asm__ volatile("movi v0.16b, #0x5a\n str q0, [%0, #32]" :: "r"(b) : "v0", "memory"); }
static void str_d_post(uint8_t *target) { uint8_t *b = target; __asm__ volatile("movi v0.16b, #0x5a\n str d0, [%0], #8" : "+r"(b) :: "v0", "memory"); }
static void stur_q(uint8_t *target) { uint8_t *b = target + 3; __asm__ volatile("movi v0.16b, #0x5a\n stur q0, [%0, #-3]" :: "r"(b) : "v0", "memory"); }
static void stlur_x(uint8_t *target) { uint64_t v = 0x1122334455667788ull; uint8_t *b = target + 3; __asm__ volatile("stlur %0, [%1, #-3]" :: "r"(v), "r"(b) : "memory"); }

static const struct { const char *name; void (*store)(uint8_t *); unsigned width; } cases[] = {
    { "str x, [xn, #imm]", str_x_uimm, 8 },
    { "str w, [xn, #imm]", str_w_uimm, 4 },
    { "strh, [xn, #imm]", strh_uimm, 2 },
    { "stur x, [xn, #-imm]", stur_x, 8 },
    { "str x, [xn], #imm", str_x_post, 8 },
    { "str x, [xn, #-imm]!", str_x_pre, 8 },
    { "str x, [xn, xm, lsl #3]", str_x_reg_lsl, 8 },
    { "str w, [xn, wm, sxtw #2]", str_w_reg_sxtw, 4 },
    { "str x, [xn, wm, uxtw]", str_x_reg_uxtw, 8 },
    { "str q, [xn, #imm]", str_q_uimm, 16 },
    { "str d, [xn], #imm", str_d_post, 8 },
    { "stur q, [xn, #-imm]", stur_q, 16 },
    { "stlur x, [xn, #-imm]", stlur_x, 8 },
};

int main(void)
{
    const size_t page = 0x4000;
    struct sigaction sa;
    unsigned boundary_reports = 0, checked = 0;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    assert((size_t)getpagesize() == page);

    for (int first_writable = 1; first_writable >= 0; first_writable--)
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
    for (unsigned before = 1; before < cases[c].width; before++)
    {
        uint8_t *map = mmap(NULL, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        uint8_t *target = map + page - before;
        assert(map != MAP_FAILED);
        assert(!mprotect(map, page, first_writable ? PROT_READ | PROT_WRITE : PROT_READ));
        assert(!mprotect(map + page, page, PROT_READ));
        faults = 0;
        cases[c].store(target);
        if (faults != 1 || seen_start != (uintptr_t)target)
        {
            fprintf(stderr, "%s, %u byte(s) before the boundary, first page %s: faults=%d fault=%#lx "
                    "start=%#lx, want %#lx\n", cases[c].name, before, first_writable ? "RW" : "RO",
                    faults, (unsigned long)seen_fault, (unsigned long)seen_start, (unsigned long)target);
            return 1;
        }
        boundary_reports += seen_fault == (uintptr_t)(map + page);
        checked++;
        munmap(map, 2 * page);
    }

    /* Not single-register stores: the fault address passes through untouched. */
    {
        uint64_t gpr[32] = { 0x10000 };
        assert(ios_store_access_start(0xf9400401u, gpr, 0x10004) == 0x10004);  /* ldr x1, [x0, #8] */
        assert(ios_store_access_start(0xf8210002u, gpr, 0x10004) == 0x10004);  /* ldadd x1, x2, [x0] */
        assert(ios_store_access_start(0xa9000801u, gpr, 0x10004) == 0x10004);  /* stp x1, x2, [x0] */
        /* A store whose access does not contain the fault address is not adjusted. */
        assert(ios_store_access_start(0xf9000401u, gpr, 0x20000) == 0x20000);  /* str x1, [x0, #8] */
    }
    printf("ok: %u page-straddling stores recovered (%u reported the page boundary as the fault address)\n",
           checked, boundary_reports);
    return boundary_reports ? 0 : 2;
}
'''

with tempfile.TemporaryDirectory() as d:
    src = Path(d) / 'check.c'
    exe = Path(d) / 'check'
    src.write_text(code)
    subprocess.run([os.environ.get('CC', 'cc'), '-std=gnu11', '-march=armv8.4-a', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
