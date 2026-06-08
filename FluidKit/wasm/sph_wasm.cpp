/**
 * sph_wasm.cpp  —  FluidKit WebAssembly SPH
 * ==========================================
 * 外部依存ゼロの軽量 SPH 流体シミュレーター。
 * Emscripten で WASM にコンパイルして Three.js から呼び出す。
 *
 * 公開 API（JavaScript 側から呼べる関数）:
 *   sph_init(n, preset)          初期化（粒子数・プリセット）
 *   sph_step(dt)                 1 ステップ進める
 *   sph_get_positions()          粒子位置の Float32Array へのポインタ
 *   sph_get_velocities()         速度スカラーの Float32Array へのポインタ
 *   sph_get_particle_count()     現在の粒子数
 *   sph_reset()                  リセット
 *   sph_set_param(key, value)    パラメータ変更（実行時調整用）
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE extern "C"
#else
#define EXPORT extern "C"
#endif

// ──────────────────────────────────────────
//  定数・設定
// ──────────────────────────────────────────

static const int MAX_N = 4096;

// ──────────────────────────────────────────
//  グローバル状態
// ──────────────────────────────────────────

static int   g_n        = 0;
static float g_h        = 0.12f;    // カーネル半径
static float g_k        = 80.0f;    // 剛性係数
static float g_rho0     = 1000.0f;  // 静止密度
static float g_mu       = 0.01f;    // 粘性
static float g_gx       = 0.0f;
static float g_gy       = -9.8f;
static float g_gz       = 0.0f;
static float g_dt       = 0.016f;
static float g_restitution = 0.25f;

// ドメイン境界 [0,1]^3
static const float BMIN = 0.0f;
static const float BMAX = 1.0f;

// 粒子データ（SoA レイアウト）
static float g_px[MAX_N], g_py[MAX_N], g_pz[MAX_N];  // 位置
static float g_vx[MAX_N], g_vy[MAX_N], g_vz[MAX_N];  // 速度
static float g_ax[MAX_N], g_ay[MAX_N], g_az[MAX_N];  // 加速度
static float g_rho[MAX_N], g_prs[MAX_N];              // 密度・圧力

// JS 返却用バッファ
static float g_pos_buf[MAX_N * 3];   // [x0,y0,z0, x1,y1,z1, ...]
static float g_spd_buf[MAX_N];       // 速度スカラー

// ──────────────────────────────────────────
//  カーネル関数（Poly6）
// ──────────────────────────────────────────

inline float kernel(float r, float h) {
    if (r >= h) return 0.0f;
    float q = 1.0f - (r * r) / (h * h);
    return q * q * q;
}

// ──────────────────────────────────────────
//  ユーティリティ
// ──────────────────────────────────────────

static float randf() {
    return (float)rand() / (float)RAND_MAX;
}

static void clampBound(int i) {
    if (g_px[i] < BMIN) { g_px[i] = BMIN; g_vx[i] *= -g_restitution; }
    if (g_px[i] > BMAX) { g_px[i] = BMAX; g_vx[i] *= -g_restitution; }
    if (g_py[i] < BMIN) { g_py[i] = BMIN; g_vy[i] *= -g_restitution; }
    if (g_py[i] > BMAX) { g_py[i] = BMAX; g_vy[i] *= -g_restitution; }
    if (g_pz[i] < BMIN) { g_pz[i] = BMIN; g_vz[i] *= -g_restitution; }
    if (g_pz[i] > BMAX) { g_pz[i] = BMAX; g_vz[i] *= -g_restitution; }
}

// ──────────────────────────────────────────
//  プリセット初期配置
// ──────────────────────────────────────────

// preset 0: 中央水滴（球形）
static void initDrop(int n) {
    srand(42);
    int placed = 0;
    while (placed < n) {
        float x = randf() - 0.5f;
        float y = randf() - 0.5f;
        float z = randf() - 0.5f;
        if (x*x + y*y + z*z <= 0.25f) {
            g_px[placed] = 0.5f + x * 0.7f;
            g_py[placed] = 0.6f + y * 0.7f;
            g_pz[placed] = 0.5f + z * 0.7f;
            g_vx[placed] = (randf()-0.5f)*0.05f;
            g_vy[placed] = (randf()-0.5f)*0.05f;
            g_vz[placed] = (randf()-0.5f)*0.05f;
            placed++;
        }
    }
}

// preset 1: 煙（下から上昇）
static void initSmoke(int n) {
    srand(99);
    g_gy = 0.8f;   // 浮力
    g_mu = 0.05f;
    g_k  = 20.0f;
    g_rho0 = 1.2f;
    for (int i = 0; i < n; i++) {
        g_px[i] = 0.25f + randf() * 0.5f;
        g_py[i] = randf() * 0.25f;
        g_pz[i] = 0.25f + randf() * 0.5f;
        g_vx[i] = (randf()-0.5f)*0.1f;
        g_vy[i] = 0.3f + randf()*0.5f;
        g_vz[i] = (randf()-0.5f)*0.1f;
    }
}

// preset 2: 左右 2 流衝突
static void initSplash(int n) {
    srand(7);
    g_k  = 100.0f;
    g_mu = 0.005f;
    int half = n / 2;
    for (int i = 0; i < n; i++) {
        float side = (i < half) ? -1.0f : 1.0f;
        g_px[i] = 0.5f + side * (0.25f + randf()*0.2f);
        g_py[i] = randf() * 0.3f;
        g_pz[i] = 0.35f + randf() * 0.3f;
        g_vx[i] = -side * (1.5f + randf());
        g_vy[i] = (randf()-0.5f)*0.05f;
        g_vz[i] = (randf()-0.5f)*0.05f;
    }
}

// ──────────────────────────────────────────
//  SPH コアステップ
// ──────────────────────────────────────────

static void sphStep(float dt) {
    // 密度・圧力
    for (int i = 0; i < g_n; i++) {
        float rho = 0.0f;
        for (int j = 0; j < g_n; j++) {
            float dx = g_px[i]-g_px[j];
            float dy = g_py[i]-g_py[j];
            float dz = g_pz[i]-g_pz[j];
            float r  = sqrtf(dx*dx + dy*dy + dz*dz);
            rho += kernel(r, g_h);
        }
        g_rho[i] = rho < 1e-6f ? 1e-6f : rho;
        g_prs[i] = g_k * (g_rho[i] - g_rho0);
    }

    // 加速度（重力 + 圧力 + 粘性）
    for (int i = 0; i < g_n; i++) {
        float ax = g_gx, ay = g_gy, az = g_gz;
        for (int j = 0; j < g_n; j++) {
            if (i == j) continue;
            float dx = g_px[i]-g_px[j];
            float dy = g_py[i]-g_py[j];
            float dz = g_pz[i]-g_pz[j];
            float r  = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-8f;
            if (r >= g_h) continue;
            float w  = kernel(r, g_h);
            // 圧力
            float pf = -(g_prs[i]+g_prs[j]) / (2.0f*g_rho[j]) * w / r;
            ax += pf * dx;
            ay += pf * dy;
            az += pf * dz;
            // 粘性
            float vf = g_mu * w / g_rho[j];
            ax += vf * (g_vx[j]-g_vx[i]);
            ay += vf * (g_vy[j]-g_vy[i]);
            az += vf * (g_vz[j]-g_vz[i]);
        }
        g_ax[i] = ax; g_ay[i] = ay; g_az[i] = az;
    }

    // 積分 (Symplectic Euler)
    for (int i = 0; i < g_n; i++) {
        g_vx[i] += g_ax[i] * dt;
        g_vy[i] += g_ay[i] * dt;
        g_vz[i] += g_az[i] * dt;
        g_px[i] += g_vx[i] * dt;
        g_py[i] += g_vy[i] * dt;
        g_pz[i] += g_vz[i] * dt;
        clampBound(i);
    }
}

// ──────────────────────────────────────────
//  公開 API
// ──────────────────────────────────────────

EXPORT void sph_init(int n, int preset) {
    g_n = (n > MAX_N) ? MAX_N : n;
    // デフォルトパラメータに戻す
    g_h = 0.12f; g_k = 80.0f; g_rho0 = 1000.0f;
    g_mu = 0.01f; g_gx = 0.0f; g_gy = -9.8f; g_gz = 0.0f;
    g_restitution = 0.25f;
    memset(g_vx,0,sizeof(float)*g_n);
    memset(g_vy,0,sizeof(float)*g_n);
    memset(g_vz,0,sizeof(float)*g_n);

    switch (preset) {
        case 1:  initSmoke(g_n);  break;
        case 2:  initSplash(g_n); break;
        default: initDrop(g_n);   break;
    }
}

EXPORT void sph_step(float dt) {
    sphStep(dt > 0.0f ? dt : g_dt);
}

EXPORT float* sph_get_positions() {
    for (int i = 0; i < g_n; i++) {
        g_pos_buf[i*3+0] = g_px[i];
        g_pos_buf[i*3+1] = g_py[i];
        g_pos_buf[i*3+2] = g_pz[i];
    }
    return g_pos_buf;
}

EXPORT float* sph_get_speeds() {
    for (int i = 0; i < g_n; i++) {
        g_spd_buf[i] = sqrtf(g_vx[i]*g_vx[i] + g_vy[i]*g_vy[i] + g_vz[i]*g_vz[i]);
    }
    return g_spd_buf;
}

EXPORT int sph_get_particle_count() { return g_n; }

EXPORT void sph_reset() { sph_init(g_n, 0); }

EXPORT void sph_set_gravity(float gx, float gy, float gz) {
    g_gx = gx; g_gy = gy; g_gz = gz;
}

EXPORT void sph_set_viscosity(float mu) { g_mu = mu; }
EXPORT void sph_set_stiffness(float k)  { g_k  = k; }
EXPORT void sph_set_kernel_radius(float h) { g_h = h; }
