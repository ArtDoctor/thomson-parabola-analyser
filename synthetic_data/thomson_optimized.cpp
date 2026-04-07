#include "thomson_shared.h"
#include <chrono>

// ============================================================
// Particle spec with cached transport constants
// ============================================================

struct ParticleSpecOpt {
    ParticleSpec base;
    double qm;    // q / m  (charge-to-mass ratio)
    double mc2;   // m * c^2  (rest energy in Joules)
};

static ParticleSpecOpt make_opt_particle(const ParticleSpec& p) {
    ParticleSpecOpt po;
    po.base = p;
    po.qm  = p.q / p.m;
    po.mc2  = p.m * C_LIGHT * C_LIGHT;
    return po;
}

// ============================================================
// Analytic transport: magnetic region (B only along x-axis)
// ============================================================
// Matching the Boris pusher sign convention:
//   dvy/dt = -(q/m) * B * vz
//   dvz/dt = +(q/m) * B * vy
//
// With omega = q*B/m:
//   vy(t) = vy0 * cos(omega*t) - vz0 * sin(omega*t)
//   vz(t) = vy0 * sin(omega*t) + vz0 * cos(omega*t)
//
// Position (integrate velocity):
//   y(t) = y0 + (vy0/w)*sin(wt) + (vz0/w)*(cos(wt) - 1)
//   z(t) = z0 + (vy0/w)*(1 - cos(wt)) + (vz0/w)*sin(wt)
//
// Note: this is NOT the same as the Lorentz-force textbook sign,
// because the Boris pusher uses a specific discrete rotation convention.
// We match it exactly here.

// Solve z(t) = z_target by Newton's method, given the analytic z(t) and vz(t).
// Returns the transit time.
static double solve_transit_B(double z0, double z_target,
                               double vy0, double vz0, double omega,
                               double t_guess) {
    double t = t_guess;
    for (int iter = 0; iter < 20; ++iter) {
        double wt = omega * t;
        double cw = std::cos(wt);
        double sw = std::sin(wt);

        // z(t) = z0 + (vy0/w)*(1 - cos(wt)) + (vz0/w)*sin(wt)
        double z_t = z0 + (vy0 / omega) * (1.0 - cw) + (vz0 / omega) * sw;
        // vz(t) = vy0*sin(wt) + vz0*cos(wt)
        double vz_t = vy0 * sw + vz0 * cw;

        double err = z_t - z_target;
        if (std::abs(err) < 1e-15) break;
        if (std::abs(vz_t) < 1e-30) break; // degenerate
        t -= err / vz_t;
    }
    return t;
}

// Apply analytic B-field transport from z0 to z_target.
// Updates r and v in place.
static void analytic_B_transport(Vec3& r, Vec3& v, double z_target,
                                  double omega) {
    double t_guess = (z_target - r.z) / v.z;
    double dt_transit = solve_transit_B(r.z, z_target, v.y, v.z, omega, t_guess);

    double wt = omega * dt_transit;
    double cw = std::cos(wt);
    double sw = std::sin(wt);

    // Final position
    double x_new = r.x + v.x * dt_transit;
    // y(t) = y0 + (vy0/w)*sin(wt) + (vz0/w)*(cos(wt) - 1)
    double y_new = r.y + (v.y / omega) * sw + (v.z / omega) * (cw - 1.0);
    // z(t) = z0 + (vy0/w)*(1 - cos(wt)) + (vz0/w)*sin(wt)
    double z_new = r.z + (v.y / omega) * (1.0 - cw) + (v.z / omega) * sw;

    // Final velocity: vy(t) = vy0*cos(wt) - vz0*sin(wt)
    double vy_new = v.y * cw - v.z * sw;
    // vz(t) = vy0*sin(wt) + vz0*cos(wt)
    double vz_new = v.y * sw + v.z * cw;

    r = {x_new, y_new, z_new};
    v.y = vy_new;
    v.z = vz_new;
    // v.x unchanged (B along x does no work)
}

// ============================================================
// Analytic transport: electric region (E in x-direction only)
// ============================================================
// E = Ex x_hat  =>  F_x = q * Ex  =>  a_x = q * Ex / m
// v_y, v_z unchanged.  v_x undergoes constant acceleration.
// x(t) = x0 + vx0 * t + 0.5 * ax * t^2
// Transit time: t = (z_target - z0) / vz  (since vz constant)

static void analytic_E_transport(Vec3& r, Vec3& v, double z_target,
                                  double ax) {
    double dt_transit = (z_target - r.z) / v.z;

    r.x = r.x + v.x * dt_transit + 0.5 * ax * dt_transit * dt_transit;
    r.y = r.y + v.y * dt_transit;
    r.z = z_target;

    v.x = v.x + ax * dt_transit;
    // v.y and v.z unchanged
}

// ============================================================
// Analytic transport: free drift
// ============================================================

static void analytic_drift(Vec3& r, Vec3& v, double z_target) {
    double dt_transit = (z_target - r.z) / v.z;

    r.x += v.x * dt_transit;
    r.y += v.y * dt_transit;
    r.z = z_target;
    // velocity unchanged
}

// ============================================================
// Relativistic transport helpers
// ============================================================

// In a pure B-field, gamma is constant (B does no work).
// The cyclotron frequency becomes omega = qB / (gamma * m).
// The analytic_B_transport formulas still apply with this omega.

// In a pure E-field along x, we need relativistic equations.
// dp_x/dt = q*Ex,  p_y = const, p_z = const
// p_x(t) = p_x0 + q*Ex*t
// gamma(t) = sqrt(1 + (p_x^2 + p_y^2 + p_z^2)/(m*c)^2)
// v_z(t) = p_z / (gamma(t) * m)  -- this changes as gamma changes!
//
// The E-field segment therefore uses stepped Boris integration so the
// changing gamma value is reflected in the detector transit accurately.

// For relativistic B-field: analytic with adjusted omega
static void analytic_B_transport_rel(Vec3& r, Vec3& v, double z_target,
                                      double q_val, double m_val) {
    double v2 = v.x*v.x + v.y*v.y + v.z*v.z;
    double gamma = 1.0 / std::sqrt(1.0 - v2 / (C_LIGHT * C_LIGHT));
    double omega = q_val * B_T / (gamma * m_val);

    double t_guess = (z_target - r.z) / v.z;
    double dt_transit = solve_transit_B(r.z, z_target, v.y, v.z, omega, t_guess);

    double wt = omega * dt_transit;
    double cw = std::cos(wt);
    double sw = std::sin(wt);

    double x_new = r.x + v.x * dt_transit;
    double y_new = r.y + (v.y / omega) * sw + (v.z / omega) * (cw - 1.0);
    double z_new = r.z + (v.y / omega) * (1.0 - cw) + (v.z / omega) * sw;

    double vy_new = v.y * cw - v.z * sw;
    double vz_new = v.y * sw + v.z * cw;

    r = {x_new, y_new, z_new};
    v.y = vy_new;
    v.z = vz_new;
}

// For relativistic E-field: use Boris stepping (gamma changes through E region)
static void boris_E_transport_rel(Vec3& r, Vec3& v, double z_target,
                                    double q_val, double m_val) {
    while (r.z < z_target) {
        double Ex = E_A;
        double Bx = 0.0;
        boris_push_rel(r, v, Ex, Bx, q_val, m_val, DT);
        if (r.z >= z_target) {
            // Interpolate back to exact boundary
            double overshoot = r.z - z_target;
            double t_back = overshoot / v.z;
            r.x -= v.x * t_back;
            r.y -= v.y * t_back;
            r.z = z_target;
            break;
        }
    }
}

// ============================================================
// Single-particle propagation with analytic sharp-boundary segments
// ============================================================

bool propagate_optimized(double K_MeV, double theta_x, double theta_y,
                          double x0, double y0,
                          const ParticleSpecOpt& part,
                          const Features& feat,
                          double& x_det, double& y_det)
{
    Vec3 r{x0, y0, 0.0};

    const double K_J = K_MeV * 1e6 * ECHARGE;
    const double qm = part.qm;
    const double q_val = part.base.q;
    const double m_val = part.base.m;
    double vmag;

    if (feat.relativistic) {
        double gamma = 1.0 + K_J / part.mc2;
        double beta = std::sqrt(1.0 - 1.0 / (gamma * gamma));
        vmag = beta * C_LIGHT;
    } else {
        vmag = std::sqrt(2.0 * K_J / m_val);
    }

    double vx = vmag * theta_x;
    double vy = vmag * theta_y;
    double vz = std::sqrt(std::max(0.0, vmag * vmag - vx * vx - vy * vy));

    Vec3 v{vx, vy, vz};

    // ---- Fringe mode: use Boris stepping throughout the continuous field ----
    if (feat.fringe) {
        for (int i = 0; i < MAX_STEPS; ++i) {
            if (r.z >= L_det) {
                double t_interp = (L_det - r.z) / v.z;
                x_det = r.x + v.x * t_interp;
                y_det = r.y + v.y * t_interp;
                return true;
            }

            double Ex, Bx;
            fields(r, Ex, Bx, true);

            if (std::abs(Ex) > 1e-6 || std::abs(Bx) > 1e-6) {
                if (feat.relativistic)
                    boris_push_rel(r, v, Ex, Bx, q_val, m_val, DT);
                else
                    boris_push_nr(r, v, Ex, Bx, qm, DT);
            } else {
                double dt_use = DT;
                if (feat.adaptive) {
                    double dz_to_field = 1e30;
                    if (r.z < B_START)
                        dz_to_field = std::min(dz_to_field, B_START - r.z);
                    else if (r.z > B_END)
                        dz_to_field = std::min(dz_to_field, r.z - B_END);
                    if (r.z < E_START)
                        dz_to_field = std::min(dz_to_field, E_START - r.z);
                    else if (r.z > E_END)
                        dz_to_field = std::min(dz_to_field, r.z - E_END);
                    double dz_to_det = L_det - r.z;
                    double safe_dist = std::min(dz_to_field, dz_to_det);
                    double step_dist = std::abs(v.z) * DT_FAST;
                    if (safe_dist > 2.0 * step_dist)
                        dt_use = DT_FAST;
                }
                r = r + dt_use * v;
            }
        }
        return false;
    }

    // ---- Analytic mode (sharp field boundaries) ----
    // The particle traverses 4 regions in order:
    //   1) Magnetic region:  z in [B_START, B_END]
    //   2) Gap (drift):      z in [B_END, E_START]
    //   3) Electric region:  z in [E_START, E_END]
    //   4) Drift to detector: z in [E_END, L_det]

    // Ensure particle starts at z=0 (B_START=0)
    // Region 1: B-field
    if (r.z < B_END && v.z > 0) {
        if (std::abs(qm) < 1e-20) {
            analytic_drift(r, v, B_END);
        } else if (feat.relativistic) {
            analytic_B_transport_rel(r, v, B_END, q_val, m_val);
        } else {
            double omega = qm * B_T;  // q * B / m
            analytic_B_transport(r, v, B_END, omega);
        }
    }

    // Region 2: Gap drift (B_END to E_START)
    if (r.z < E_START && v.z > 0) {
        analytic_drift(r, v, E_START);
    }

    // Region 3: E-field
    if (r.z < E_END && v.z > 0) {
        if (std::abs(qm) < 1e-20) {
            analytic_drift(r, v, E_END);
        } else if (feat.relativistic) {
            boris_E_transport_rel(r, v, E_END, q_val, m_val);
        } else {
            double ax = qm * E_A;  // q * E / m
            analytic_E_transport(r, v, E_END, ax);
        }
    }

    // Region 4: Drift to detector
    if (v.z > 0) {
        analytic_drift(r, v, L_det);
        x_det = r.x;
        y_det = r.y;
        return true;
    }

    return false;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    Features feat;
    std::vector<ParticleSpec> particles_raw;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-relativistic") {
            feat.relativistic = true;
        } else if (arg == "-fringe") {
            feat.fringe = true;
        } else if (arg == "-adaptive") {
            feat.adaptive = true;
        } else if (arg == "-beam") {
            feat.beam = true;
        } else if (arg == "-particle") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -particle requires an argument\n";
                return 1;
            }
            particles_raw.push_back(parse_particle(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Default particle: C1
    if (particles_raw.empty())
        particles_raw.push_back(parse_particle("C1"));

    // Cache particle constants used repeatedly during transport.
    std::vector<ParticleSpecOpt> particles;
    for (const auto& p : particles_raw)
        particles.push_back(make_opt_particle(p));

    // Print config to stderr
    std::cerr << "Features:";
    if (feat.relativistic) std::cerr << " relativistic";
    if (feat.fringe)       std::cerr << " fringe";
    if (feat.adaptive)     std::cerr << " adaptive";
    if (feat.beam)         std::cerr << " beam";
    if (!feat.relativistic && !feat.fringe && !feat.adaptive && !feat.beam)
        std::cerr << " none (original mode)";
    std::cerr << "\nParticles:";
    for (auto& p : particles)
        std::cerr << " " << p.base.label << "(q=" << p.base.charge_num
                  << "e, m=" << p.base.mass_amu << " amu)";
    std::cerr << "\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Simulate each particle type
    for (const auto& part : particles) {
        // ---- TNSA energy scaling ----
        // K_max scales with charge state: K_max(Z) = Z * Phi
        int Z = std::abs(part.base.charge_num);
        int Z_eff = (Z == 0) ? 1 : Z;
        double K_max_MeV = Z_eff * SHEATH_PHI_MV;
        double K_min_MeV = K_MIN_MEV;
        if (K_min_MeV >= K_max_MeV) K_min_MeV = 0.1 * K_max_MeV;

        // Particle count scales as N / Z^COUNT_POWER
        int N_part = static_cast<int>(N_PARTICLES / std::pow(Z_eff, COUNT_POWER));
        if (Z == 0) N_part = N_PARTICLES / 2; // bright spreaded spot
        if (N_part < 1000) N_part = 1000;  // minimum for statistics

        // Exponential spectrum via inverse CDF:
        //   dN/dK ~ exp(-K/kT), K in [K_min, K_max]
        //   CDF(K) = [exp(-K_min/kT) - exp(-K/kT)] / [exp(-K_min/kT) - exp(-K_max/kT)]
        //   Inverse: K = -kT * ln(exp(-K_min/kT) - u * [exp(-K_min/kT) - exp(-K_max/kT)])
        double exp_min = std::exp(-K_min_MeV / KT_MEV / std::pow(Z_eff, ENERGY_EXPONENT));
        double exp_max = std::exp(-K_max_MeV / KT_MEV / std::pow(Z_eff, ENERGY_EXPONENT));
        double exp_range = exp_min - exp_max;

        // Per-species angular divergence: wider for lower charge states
        double spread_z = SPREAD_EXTRA / Z_eff;
        if (Z == 0) spread_z *= 2.0; // make neutral spot extra spreaded
        double sigma_ang = std::sqrt(ANGLE_SIGMA * ANGLE_SIGMA + spread_z * spread_z);

        std::cerr << part.base.label
                  << ": Z=" << Z
                  << ", K=[" << K_min_MeV << ", " << K_max_MeV << "] MeV"
                  << ", N=" << N_part
                  << ", sigma=" << sigma_ang * 1e3 << " mrad\n";

        std::vector<double> X, Y, K;
        std::vector<std::string> labels;
        X.reserve(N_part);
        Y.reserve(N_part);
        K.reserve(N_part);

        #pragma omp parallel
        {
            std::mt19937_64 rng(1234 + omp_get_thread_num());
            std::normal_distribution<double> ang(0.0, sigma_ang);
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            std::normal_distribution<double> beam_pos(0.0, BEAM_SIGMA);

            // Reserve thread-local buffers to reduce repeated allocations.
            int chunk = N_part / omp_get_num_threads() + 1;
            std::vector<double> Xloc, Yloc, Kloc;
            Xloc.reserve(chunk);
            Yloc.reserve(chunk);
            Kloc.reserve(chunk);

            #pragma omp for
            for (int i = 0; i < N_part; ++i) {
                // Exponential energy sampling (inverse CDF)
                double u = u01(rng);
                double K_MeV = -KT_MEV * std::pow(Z_eff, ENERGY_EXPONENT) * std::log(exp_min - u * exp_range);

                double tx = ang(rng);
                double ty = ang(rng);

                double x0 = 0.0, y0 = 0.0;
                if (feat.beam) {
                    x0 = beam_pos(rng);
                    y0 = beam_pos(rng);
                }

                double xd, yd;
                if (propagate_optimized(K_MeV, tx, ty, x0, y0,
                                         part, feat, xd, yd)) {
                    Xloc.push_back(xd);
                    Yloc.push_back(yd);
                    Kloc.push_back(K_MeV);
                }
            }

            #pragma omp critical
            {
                X.insert(X.end(), Xloc.begin(), Xloc.end());
                Y.insert(Y.end(), Yloc.begin(), Yloc.end());
                K.insert(K.end(), Kloc.begin(), Kloc.end());
            }
        }

        std::cerr << part.base.label << " hits: " << X.size() << "\n";

        for (size_t i = 0; i < X.size(); ++i)
            std::cout << -Y[i] << " " << X[i] << " " << K[i]
                      << " " << part.base.label << "\n";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cerr << "Elapsed: " << elapsed << " s\n";

    return 0;
}
