//! Embedded PTX source for matrix multiplication kernel

pub const PTX_SOURCE: &str = r#".version 6.0
.target sm_50
.address_size 64

.visible .entry matrix_multiply(
    .param .u64 matrix_multiply_param_0,
    .param .u64 matrix_multiply_param_1,
    .param .u64 matrix_multiply_param_2,
    .param .u32 matrix_multiply_param_3,
    .param .u32 matrix_multiply_param_4,
    .param .u32 matrix_multiply_param_5
)
{
    .reg .pred      %p<3>;
    .reg .b32       %r<14>;
    .reg .f64       %fd<5>;
    .reg .b64       %rd<13>;

    ld.param.u64    %rd1, [matrix_multiply_param_0];
    ld.param.u64    %rd2, [matrix_multiply_param_1];
    ld.param.u64    %rd3, [matrix_multiply_param_2];
    ld.param.u32    %r2, [matrix_multiply_param_3];
    ld.param.u32    %r3, [matrix_multiply_param_4];
    ld.param.u32    %r4, [matrix_multiply_param_5];
    
    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r1, %r6, %r5, %r7;
    
    mov.u32         %r8, %ctaid.y;
    mov.u32         %r9, %ntid.y;
    mov.u32         %r10, %tid.y;
    mad.lo.s32      %r11, %r9, %r8, %r10;
    
    setp.ge.s32     %p1, %r11, %r2;
    setp.ge.s32     %p2, %r1, %r4;
    or.pred         %p1, %p1, %p2;
    @%p1 bra        LBB0_5;

    mov.b64         %rd4, 0;
    cvt.rn.f64.s64  %fd1, %rd4;
    mov.u32         %r12, 0;

LBB0_2:
    setp.ge.s32     %p2, %r12, %r3;
    @%p2 bra        LBB0_4;

    mad.lo.s32      %r13, %r11, %r3, %r12;
    mul.wide.s32    %rd4, %r13, 8;
    add.s64         %rd5, %rd1, %rd4;
    ld.global.f64   %fd2, [%rd5];
    
    mad.lo.s32      %r13, %r12, %r4, %r1;
    mul.wide.s32    %rd6, %r13, 8;
    add.s64         %rd7, %rd2, %rd6;
    ld.global.f64   %fd3, [%rd7];
    
    fma.rn.f64      %fd1, %fd2, %fd3, %fd1;
    add.s32         %r12, %r12, 1;
    bra.uni         LBB0_2;

LBB0_4:
    mad.lo.s32      %r13, %r11, %r4, %r1;
    mul.wide.s32    %rd8, %r13, 8;
    add.s64         %rd9, %rd3, %rd8;
    st.global.f64   [%rd9], %fd1;

LBB0_5:
    ret;
}

.visible .entry init_rng(
    .param .u64 init_rng_param_0,
    .param .u64 init_rng_param_1
)
{
    .reg .b64       %rd<4>;

    ld.param.u64    %rd1, [init_rng_param_0];
    ld.param.u64    %rd2, [init_rng_param_1];
    st.global.u64   [%rd1], %rd2;
    ret;
}

.visible .entry generate_random(
    .param .u64 generate_random_param_0,
    .param .u64 generate_random_param_1,
    .param .u32 generate_random_param_2
)
{
    .reg .pred      %p<2>;
    .reg .b32       %r<7>;
    .reg .f64       %fd<3>;
    .reg .b64       %rd<10>;

    ld.param.u64    %rd1, [generate_random_param_0];
    ld.param.u64    %rd2, [generate_random_param_1];
    ld.param.u32    %r1, [generate_random_param_2];
    
    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r5, %r2, %r3, %r4;
    
    setp.ge.u32     %p1, %r5, %r1;
    @%p1 bra        LBB2_2;

    ld.global.u64   %rd3, [%rd2];
    cvt.u64.u32     %rd4, %r5;
    add.s64         %rd3, %rd3, %rd4;
    
    mov.b64         %rd5, 1664525;
    mul.lo.s64      %rd6, %rd3, %rd5;
    mov.b64         %rd7, 1013904223;
    add.s64         %rd6, %rd6, %rd7;
    
    mov.b64         %rd8, 0x7FFFFFFFFFFFFFFF;
    and.b64         %rd6, %rd6, %rd8;
    cvt.rn.f64.u64  %fd1, %rd6;
    cvt.rn.f64.u64  %fd2, %rd8;
    div.rn.f64      %fd1, %fd1, %fd2;
    
    mul.wide.u32    %rd9, %r5, 8;
    add.s64         %rd9, %rd1, %rd9;
    st.global.f64   [%rd9], %fd1;

LBB2_2:
    ret;
}"#;
