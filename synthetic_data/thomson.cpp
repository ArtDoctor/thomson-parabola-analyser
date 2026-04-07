#include "thomson_shared.h"

// ============================================================
// Single particle propagation
// ============================================================

bool propagate(double K_MeV, double theta_x, double theta_y,
               double x0, double y0,
               double q_val, double m_val,
               const Features& feat,
               double& x_det, double& y_det)
{
    Vec3 r{x0, y0, 0.0};

    const double K_J = K_MeV * 1e6 * ECHARGE;
    double vmag;

    if (feat.relativistic) {
        // Relativistic: K = (gamma - 1) * m * c^2
        // gamma = 1 + K / (m * c^2)
        double gamma = 1.0 + K_J / (m_val * C_LIGHT * C_LIGHT);
        double beta = std::sqrt(1.0 - 1.0 / (gamma * gamma));
        vmag = beta * C_LIGHT;
    } else {
        vmag = std::sqrt(2.0 * K_J / m_val);
    }

    double vx = vmag * theta_x;
    double vy = vmag * theta_y;
    double vz = std::sqrt(std::max(0.0, vmag * vmag - vx * vx - vy * vy));

    Vec3 v{vx, vy, vz};

    const double qm = q_val / m_val;

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (r.z >= L_det) {
            double t_interp = (L_det - r.z) / v.z;
            x_det = r.x + v.x * t_interp;
            y_det = r.y + v.y * t_interp;
            return true;
        }

        double Ex, Bx;
        fields(r, Ex, Bx, feat.fringe);

        if (std::abs(Ex) > 1e-6 || std::abs(Bx) > 1e-6) {
            if (feat.relativistic)
                boris_push_rel(r, v, Ex, Bx, q_val, m_val, DT);
            else
                boris_push_nr(r, v, Ex, Bx, qm, DT);
        } else {
            // Free drift; optionally stretch the step when far from field regions.
            double dt_use = DT;

            if (feat.adaptive) {
                double dz_to_field = 1e30;

                // Distance to B-field region
                if (r.z < B_START)
                    dz_to_field = std::min(dz_to_field, B_START - r.z);
                else if (r.z > B_END)
                    dz_to_field = std::min(dz_to_field, r.z - B_END);

                // Distance to E-field region
                if (r.z < E_START)
                    dz_to_field = std::min(dz_to_field, E_START - r.z);
                else if (r.z > E_END)
                    dz_to_field = std::min(dz_to_field, r.z - E_END);

                // Also consider distance to detector
                double dz_to_det = L_det - r.z;

                double safe_dist = std::min(dz_to_field, dz_to_det);

                // If safe distance allows, use bigger step
                // Estimate distance traveled in one fast step
                double step_dist = std::abs(v.z) * DT_FAST;
                if (safe_dist > 2.0 * step_dist)
                    dt_use = DT_FAST;
            }

            r = r + dt_use * v;
        }
    }

    return false;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    Features feat;
    std::vector<ParticleSpec> particles;

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
            particles.push_back(parse_particle(argv[++i]));
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
    if (particles.empty())
        particles.push_back(parse_particle("C1"));

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
        std::cerr << " " << p.label << "(q=" << p.charge_num
                  << "e, m=" << p.mass_amu << " amu)";
    std::cerr << "\n";

    // Simulate each particle type
    for (const auto& part : particles) {
        // ---- TNSA energy scaling ----
        int Z = std::abs(part.charge_num);
        int Z_eff = (Z == 0) ? 1 : Z;
        double K_max_MeV = Z_eff * SHEATH_PHI_MV;
        double K_min_MeV = K_MIN_MEV;
        if (K_min_MeV >= K_max_MeV) K_min_MeV = 0.1 * K_max_MeV;

        int N_part = static_cast<int>(N_PARTICLES / std::pow(Z_eff, COUNT_POWER));
        if (Z == 0) N_part = N_PARTICLES / 2;
        if (N_part < 1000) N_part = 1000;

        double exp_min = std::exp(-K_min_MeV / KT_MEV);
        double exp_max = std::exp(-K_max_MeV / KT_MEV);
        double exp_range = exp_min - exp_max;

        // Per-species angular divergence: wider for lower charge states
        double spread_z = SPREAD_EXTRA / Z_eff;
        if (Z == 0) spread_z *= 2.0;
        double sigma_ang = std::sqrt(ANGLE_SIGMA * ANGLE_SIGMA + spread_z * spread_z);

        std::cerr << part.label
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

            std::vector<double> Xloc, Yloc, Kloc;

            #pragma omp for
            for (int i = 0; i < N_part; ++i) {
                double u = u01(rng);
                double K_MeV = -KT_MEV * std::log(exp_min - u * exp_range);

                double tx = ang(rng);
                double ty = ang(rng);

                double x0 = 0.0, y0 = 0.0;
                if (feat.beam) {
                    x0 = beam_pos(rng);
                    y0 = beam_pos(rng);
                }

                double xd, yd;
                if (propagate(K_MeV, tx, ty, x0, y0,
                              part.q, part.m, feat, xd, yd)) {
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

        std::cerr << part.label << " hits: " << X.size() << "\n";

        for (size_t i = 0; i < X.size(); ++i)
            std::cout << -Y[i] << " " << X[i] << " " << K[i]
                      << " " << part.label << "\n";
    }

    return 0;
}
