#ifndef THOMSON_SHARED_H
#define THOMSON_SHARED_H

#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <omp.h>

// ============================================================
// Physical constants
// ============================================================

constexpr double ECHARGE = 1.602176634e-19;    // C
constexpr double m_p     = 1.660539e-27;   // unified atomic mass unit, kg
constexpr double C_LIGHT = 299792458;        // m/s

// ============================================================
// Simulation parameters
// ============================================================

constexpr double DT        = 5e-13;
constexpr int    MAX_STEPS  = 2000000;

// Geometry
constexpr double cm  = 1e-2;
constexpr double LiB = 8 * cm;
constexpr double LiE = 8 * cm;
constexpr double L_det = LiB + 29.5 * cm;
constexpr double LfB   = L_det - LiB;
constexpr double LfE   = LfB - LiE;

// Fields
constexpr double B_T = 0.14536;   // Tesla; same as utils.physics.B_T
constexpr double E_A = 0.13e6;    // V/m

// Field boundaries
constexpr double B_START = 0.0;
constexpr double B_END   = LiB;
constexpr double E_START = LiB + 1 * cm;
constexpr double E_END   = LiB + 1 * cm + LiE;

// Fringe field parameter
constexpr double FRINGE_DELTA = 0.5 * cm;

// Energy scaling (TNSA model)
// Reference: calibrated to C1+ experimental image.
// In TNSA, all ions see the same sheath potential Phi.
// K_max(Z) = Z * e * Phi, so K_max scales linearly with charge state.
// The energy spectrum is exponential: dN/dK ~ exp(-K/kT), not uniform.
constexpr double K_MIN_MEV      = 0.010;   // detector low-energy cutoff (all species)
constexpr double K_MAX_REF_MEV  = 1.889;   // K_max for reference species (C1+, Z=1)
constexpr int    Z_REF          = 1;        // reference charge state
constexpr double SHEATH_PHI_MV  = K_MAX_REF_MEV / Z_REF;  // sheath potential in MV
constexpr double KT_MEV         = 0.4;   // exponential temperature (matched to C1+ mean)

// Particle count scaling: higher charge states are slightly rarer.
// Mild falloff: N(Z) ~ N_ref / Z^0.3
constexpr double COUNT_POWER    = 0.1;     // exponent for count scaling

// Beam divergence scaling: lower charge states have broader source
// Effective sigma_angle = sqrt(ANGLE_SIGMA^2 + (SPREAD_EXTRA/Z)^2)
// This makes inner (low q/m) parabolas visibly wider.
constexpr double SPREAD_EXTRA   = 0.001;   // 3 mrad / Z  additional divergence
constexpr double ENERGY_EXPONENT = 0.30;     // exponent for energy scaling

// Beam
constexpr double ANGLE_SIGMA = 0.0005;
constexpr double BEAM_SIGMA  = 0.5e-3;  // 0.5 mm
constexpr int    N_PARTICLES = 200000;

// Adaptive stepping
constexpr double DT_FAST       = 10.0 * DT;
constexpr double FIELD_MARGIN  = 5.0 * DT_FAST * 2e5; // ~5 fast steps * typical v

// ============================================================
// Particle specification
// ============================================================

struct ParticleSpec {
    std::string label;   // e.g. "C4", "H", "e"
    std::string symbol;  // e.g. "C", "H", "e", "Si", "O"
    int    charge_num;   // signed: +1, +4, -1 for electron, etc.
    double mass_amu;     // mass in amu
    double q;            // charge in coulombs
    double m;            // mass in kg
};

// Returns mass number A so the simulator can convert amu into kilograms.
static double lookup_mass_number(const std::string& sym) {
    if (sym == "C")  return 12.0;
    if (sym == "H")  return 1.0;
    if (sym == "He") return 4.0;
    if (sym == "Li") return 7.0;
    if (sym == "Be") return 9.0;
    if (sym == "B")  return 11.0;
    if (sym == "N")  return 14.0;
    if (sym == "O")  return 16.0;
    if (sym == "F")  return 19.0;
    if (sym == "Ne") return 20.0;
    if (sym == "Si") return 28.0;
    if (sym == "P")  return 31.0;
    if (sym == "S")  return 32.0;
    if (sym == "Cl") return 35.0;
    if (sym == "Ar") return 40.0;
    if (sym == "Fe") return 56.0;
    if (sym == "e")  return 5.485799090e-4;  // electron mass in units of m_p
    return -1.0;  // unknown
}

static ParticleSpec parse_particle(const std::string& spec) {
    ParticleSpec p;
    p.label = spec;

    // Find where digits start
    size_t digit_start = spec.size();
    for (size_t i = 0; i < spec.size(); ++i) {
        if (std::isdigit(spec[i])) {
            digit_start = i;
            break;
        }
        // Allow negative sign only at digit boundary
        if (spec[i] == '-' && i + 1 < spec.size() && std::isdigit(spec[i + 1])) {
            digit_start = i;
            break;
        }
    }

    p.symbol = spec.substr(0, digit_start);

    if (digit_start < spec.size()) {
        p.charge_num = std::stoi(spec.substr(digit_start));
    } else {
        // Default charge: -1 for electron, +1 for everything else
        p.charge_num = (p.symbol == "e") ? -1 : 1;
    }

    p.mass_amu = lookup_mass_number(p.symbol);
    if (p.mass_amu < 0.0) {
        std::cerr << "Unknown particle symbol: " << p.symbol << "\n";
        std::exit(1);
    }

    p.q = p.charge_num * ECHARGE;
    p.m = p.mass_amu * m_p;

    return p;
}

// ============================================================
// Feature flags
// ============================================================

struct Features {
    bool relativistic = false;
    bool fringe       = false;
    bool adaptive     = false;
    bool beam         = false;
};

// ============================================================
// Simple 3D vector
// ============================================================

struct Vec3 {
    double x, y, z;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator*(double s, const Vec3& v) {
    return {s * v.x, s * v.y, s * v.z};
}

// ============================================================
// Fields
// ============================================================

inline void fields(const Vec3& r, double& Ex, double& Bx, bool fringe) {
    Ex = 0.0;
    Bx = 0.0;

    if (fringe) {
        // Smooth tanh profile
        Bx = B_T * 0.5 * (std::tanh((r.z - B_START) / FRINGE_DELTA)
                         - std::tanh((r.z - B_END)   / FRINGE_DELTA));
        Ex = E_A * 0.5 * (std::tanh((r.z - E_START) / FRINGE_DELTA)
                         - std::tanh((r.z - E_END)   / FRINGE_DELTA));
    } else {
        // Piecewise-constant field profile with sharp region boundaries.
        if (r.z >= 0.0 && r.z <= LiB)
            Bx = B_T;

        if (r.z >= LiB + 1 * cm && r.z <= LiB + 1 * cm + LiE)
            Ex = E_A;
    }
}

// ============================================================
// Boris pusher — non-relativistic
// ============================================================

inline void boris_push_nr(Vec3& r, Vec3& v, double Ex, double Bx,
                           double qm, double dt) {
    // Half E-kick
    v.x += 0.5 * qm * Ex * dt;

    // Rotation around x-axis
    const double t = 0.5 * qm * Bx * dt;
    const double s = 2.0 * t / (1.0 + t * t);

    double vy = v.y - v.z * t;
    double vz = v.z + v.y * t;

    v.y -= vz * s;
    v.z += vy * s;

    // Half E-kick
    v.x += 0.5 * qm * Ex * dt;

    // Position update
    r = r + dt * v;
}

// ============================================================
// Boris pusher — relativistic
// ============================================================

inline void boris_push_rel(Vec3& r, Vec3& v, double Ex, double Bx,
                            double q_val, double m_val, double dt) {
    // Work with momentum u = gamma * v
    double gamma = 1.0 / std::sqrt(1.0 - (v.x*v.x + v.y*v.y + v.z*v.z)
                                           / (C_LIGHT * C_LIGHT));
    double ux = gamma * v.x;
    double uy = gamma * v.y;
    double uz = gamma * v.z;

    double qdt2m = q_val * dt / (2.0 * m_val);

    // Half E-kick (E is in x-direction)
    ux += qdt2m * Ex;

    // Update gamma at half step
    double u2 = ux*ux + uy*uy + uz*uz;
    double gamma_half = std::sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT));

    // Rotation: t-vector (B is in x-direction)
    double tx = qdt2m * Bx / gamma_half;

    double uy_prime = uy - uz * tx;
    double uz_prime = uz + uy * tx;

    double s_factor = 2.0 * tx / (1.0 + tx * tx);

    uy -= uz_prime * s_factor;
    uz += uy_prime * s_factor;

    // Half E-kick
    ux += qdt2m * Ex;

    // Recover velocity from momentum
    u2 = ux*ux + uy*uy + uz*uz;
    gamma = std::sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT));
    v.x = ux / gamma;
    v.y = uy / gamma;
    v.z = uz / gamma;

    // Position update
    r = r + dt * v;
}

// ============================================================
// CLI print_usage
// ============================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  -relativistic   Enable relativistic Boris pusher\n"
              << "  -fringe         Enable fringe fields (tanh profile)\n"
              << "  -adaptive       Enable adaptive time stepping\n"
              << "  -beam           Enable finite-size beam (Gaussian x,y)\n"
              << "  -particle SPEC  Add particle (e.g. C4, H, e, Si2, O6)\n"
              << "                  Can be repeated. Default: C1\n";
}

#endif // THOMSON_SHARED_H
