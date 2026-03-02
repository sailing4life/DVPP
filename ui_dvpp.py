#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for the IMOCA 60 DVPP.

Run with:
    streamlit run ui_dvpp.py

Tabs
----
1. Panel Solver   — compute A(ω), B(ω), A_inf, C from the hull STL
2. Simulation     — set wind conditions and run the 6-DOF time-domain sim
3. Results        — velocity, position, angles, and force decomposition
4. Coefficients   — inspect hydrodynamic matrices from the panel solver
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, time, tempfile

# ── project modules ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_solver   import PanelSolver
from simulink_dvpp import SIMULINK_DISPLACEMENT_KG, run_simulation
from simulink_dvpp.radiation import DEFAULT_OMEGA as _NEMOH_OMEGA, DEFAULT_B33 as _NEMOH_B33
from orc_dxt import parse_dxt
from validate_sphere import _retardation_kernel, _simulate_cummins, _analytical_drop
from simulink_dvpp.radiation import A_INF as _NEMOH_A_INF

try:
    from capytaine_solver import CapytaineSolver, _CAPYTAINE_OK, capytaine_available
except ImportError:
    _CAPYTAINE_OK = False
    capytaine_available = lambda python_executable=None: False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMOCA 60 DVPP",
    page_icon="⛵",
    layout="wide",
)
st.title("IMOCA 60 — Dynamic Vessel Performance Predictor")

# Session state init
for key, default in [
    ('solver_done',  False),
    ('solver',       None),
    ('sim_done',     False),
    ('t_vals',       None),
    ('y_vals',       None),
    ('sim_outputs',  None),
    ('mass_kg',      float(SIMULINK_DISPLACEMENT_KG)),
    ('drop_done',    False),
    ('drop_results', None),
    ('capytaine_python', os.environ.get('CAPYTAINE_PYTHON', '')),
    ('orc_rig',      None),
    ('jib_id',       None),
    ('headsail_id',  None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔬 Panel Solver", "▶️ Simulation", "📈 Results", "📊 Coefficients", "🏊 Drop Test"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PANEL SOLVER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Panel Solver — Strip Theory BEM")
    st.markdown(
        """
        Computes **A_inf**, **B(ω)**, **A(ω)** and the hydrostatic restoring
        matrix **C** directly from the hull STL.

        *Theory: Salvesen, Tuck & Faltinsen (1970) strip theory;
        Vossers (1960) damping integrals; Ogilvie (1964) Kramers–Kronig
        for A(ω); Lewis (1929) form for infinite-frequency added mass.*
        """
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Settings")

        # STL file — upload or use default
        uploaded = st.file_uploader("Upload hull STL (body frame, z up)", type="stl")
        default_stl = os.path.join(os.path.dirname(__file__), "PRB F.stl")
        use_default = st.checkbox("Use default hull (PRB F.stl)", value=True)

        st.markdown("**BEM solver**")
        st.session_state.capytaine_python = st.text_input(
            "External Capytaine Python",
            value=st.session_state.capytaine_python,
            help="Optional path to a separate Python executable where Capytaine "
                 "works. Leave empty to try the current interpreter."
        )
        external_python = st.session_state.capytaine_python.strip() or None
        capy_available = capytaine_available(external_python)

        if external_python:
            if capy_available:
                st.caption(f"Using Capytaine from: `{external_python}`")
            else:
                st.warning("The configured external Python could not run a "
                           "Capytaine probe solve.")
        elif _CAPYTAINE_OK:
            st.caption("Capytaine is available in the current interpreter.")

        if capy_available:
            solver_choice = st.radio(
                "Solver type",
                ["Strip Theory (fast)", "3D BEM — Capytaine (accurate)"],
                horizontal=True,
                help="Strip theory is ~100× faster but underestimates damping for "
                     "non-slender bodies.  Capytaine solves the full 3-D radiation "
                     "problem — same method as NEMOH."
            )
        else:
            solver_choice = "Strip Theory (fast)"
            st.info("3D BEM unavailable here. Provide an external Capytaine "
                    "Python above or use strip theory.")

        use_capytaine = solver_choice.startswith("3D")

        if not use_capytaine:
            n_sections = st.slider("Number of cross-sections", 10, 60, 30, step=5)
        omega_min  = st.number_input("ω min [rad/s]", value=0.10, step=0.05)
        omega_max  = st.number_input("ω max [rad/s]", value=5.00, step=0.10)
        n_omega    = st.slider("Number of frequencies", 10, 80, 40,
                               help="More points = smoother retardation kernel.  "
                                    "40+ recommended for drop tests.")

        rho_water  = st.number_input("Water density ρ [kg/m³]", value=1025.0, step=1.0)

        st.markdown("---")
        st.markdown("**Hydrostatic equilibrium**")
        mass_kg = st.number_input(
            "Vessel displacement [kg]",
            value=int(SIMULINK_DISPLACEMENT_KG), step=100,
            help="Total vessel mass (hull + keel bulb + ballast). "
                 "Used to solve buoyancy = weight for the design waterline."
        )
        override_wl = st.checkbox("Override waterline z manually", value=False)
        waterline_z = None
        if override_wl:
            waterline_z = st.number_input(
                "Waterline z [m] in STL frame",
                value=0.70, step=0.01,
                help="Overrides the hydrostatic equilibrium calculation. "
                     "Use only if you know the exact waterline z in the STL."
            )
        else:
            st.caption("Waterline solved from buoyancy = weight (shown in terminal).")

        if st.button("🔬 Run Panel Solver"):
            # Determine STL path
            if uploaded is not None:
                # Write to temp file
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                    f.write(uploaded.read())
                    stl_path = f.name
            elif use_default and os.path.exists(default_stl):
                stl_path = default_stl
            else:
                st.error("Please upload an STL or check the default file path.")
                st.stop()

            omega = np.linspace(omega_min, omega_max, n_omega)

            label = "3D BEM (Capytaine) …" if use_capytaine else "strip theory …"
            with st.spinner(f"Running {label}"):
                try:
                    if use_capytaine:
                        solver = CapytaineSolver(stl_path, rho=rho_water,
                                                 mass=float(mass_kg),
                                                 waterline_z=waterline_z,
                                                 python_executable=external_python)
                        solver.run(omega=omega)
                    else:
                        solver = PanelSolver(stl_path, rho=rho_water,
                                             mass=float(mass_kg),
                                             waterline_z=waterline_z)
                        solver.run(omega=omega, n_sections=n_sections)
                    st.session_state.solver      = solver
                    st.session_state.solver_done = True
                    st.session_state.mass_kg     = float(mass_kg)
                    st.success(f"Solver complete ({solver_choice}).")
                except Exception as e:
                    st.error(f"Solver error: {e}")
                    import traceback; st.code(traceback.format_exc())

    with col_r:
        if st.session_state.solver_done:
            solver = st.session_state.solver

            if solver.sections is not None:
                segs_valid = [s for s in solver.sections if s is not None]
                x_arr  = np.array([s['x']    for s in segs_valid])
                b_arr  = np.array([s['b_wl'] for s in segs_valid])
                T_arr  = np.array([s['T']    for s in segs_valid])

                st.subheader("Hull cross-section geometry")
                fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                axes[0].plot(x_arr, 2*b_arr, label='Beam B(x)')
                axes[0].plot(x_arr, T_arr,   label='Draft T(x)', linestyle='--')
                axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('[m]')
                axes[0].set_title('Waterplane breadth & draft'); axes[0].legend(); axes[0].grid()

                axes[1].plot(x_arr, [s['A_s'] for s in segs_valid], color='teal')
                axes[1].set_xlabel('x [m]'); axes[1].set_ylabel('A_s [m²]')
                axes[1].set_title('Submerged section area'); axes[1].grid()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("3D BEM — cross-section geometry not available for panel methods.")

            st.subheader("Hydrostatic restoring matrix C")
            C_diag = np.diag(solver.C)
            labels = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']
            for i, (lbl, val) in enumerate(zip(labels, C_diag)):
                if abs(val) > 1e-3:
                    st.write(f"C[{lbl}] = {val:,.1f} N/m or N·m/rad")

            st.subheader("A_inf diagonal [kg]")
            A_diag = np.diag(solver.A_inf)
            for i, (lbl, val) in enumerate(zip(labels, A_diag)):
                if abs(val) > 1:
                    st.write(f"A_inf[{lbl}] = {val:,.0f}")
        else:
            st.info("Run the panel solver to see results here.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("6-DOF Time-Domain Simulation")

    sim_default_hull = os.path.join(os.path.dirname(__file__), "PRB F.stl")
    wave_mode_labels = {
        "Calm water": "calm",
        "Regular wave (wind-based)": "regular",
        "Irregular waves (3 components)": "irregular_components",
        "Irregular waves (spectrum)": "irregular_spectrum",
    }

    col_wind, col_sail, col_app2, col_time = st.columns(4)

    with col_wind:
        st.subheader("Wind")
        TWA = st.slider("True Wind Angle [°]", 0, 180, 70)
        TWS = st.slider("True Wind Speed [kn]", 0, 40, 15)

        st.markdown("**Waves**")
        wave_mode_label = st.selectbox(
            "Wave model",
            list(wave_mode_labels.keys()),
            index=0,
        )
        wave_mode = wave_mode_labels[wave_mode_label]
        wave_ramp = st.slider("Wave ramp", 0.0, 1.0, 0.0, step=0.05)
        wave_angle_deg = st.slider("Wave heading [°]", 0, 180, 180)
        wave_wind_speed_kn = st.slider("Wave wind speed [kn]", 0, 50, 15)

    with col_sail:
        st.subheader("Sail trim")
        Reef = st.slider("Reef factor", 0.0, 1.0, 1.0, step=0.05)
        Flat = st.slider("Flat factor", 0.0, 1.0, 1.0, step=0.05)
        C0_enabled = st.checkbox("Flying headsail (C0)", value=False)

        st.markdown("---")
        st.markdown("**ORC sail inventory**")
        dxt_file = st.file_uploader("Upload DXT certificate", type=["dxt", "xml"],
                                    help="ORC DXT certificate — extracts actual sail dimensions.")
        if dxt_file is not None:
            try:
                rig = parse_dxt(dxt_file)
                st.session_state.orc_rig = rig
                # Reset selections when a new file is loaded
                st.session_state.jib_id = None
                st.session_state.headsail_id = None
                st.caption(f"Loaded: **{rig.yacht_name}** (cert {rig.cert_no})")
            except Exception as e:
                st.error(f"Could not parse DXT: {e}")

        rig = st.session_state.orc_rig
        jib_geom = None
        headsail_geom = None
        mainsail_geom = None

        if rig is not None:
            mainsail_geom = rig.mainsail

            jibs = rig.jibs()
            if jibs:
                jib_labels = [h.label() for h in jibs]
                default_jib = (
                    next((i for i, h in enumerate(jibs) if h.sail_id == st.session_state.jib_id), 0)
                )
                sel_jib = st.selectbox("Non-flying jib", jib_labels, index=default_jib)
                chosen_jib = jibs[jib_labels.index(sel_jib)]
                st.session_state.jib_id = chosen_jib.sail_id
                jib_geom = chosen_jib
                st.caption(f"Area: {chosen_jib.sail_area:.1f} m²  ·  Luff: {chosen_jib.JIBLUFF:.2f} m")

            flying = rig.flying_headsails()
            if flying:
                fly_labels = [h.label() for h in flying]
                default_fly = (
                    next((i for i, h in enumerate(flying) if h.sail_id == st.session_state.headsail_id), 0)
                )
                sel_fly = st.selectbox("Flying headsail / gennaker", fly_labels, index=default_fly)
                chosen_fly = flying[fly_labels.index(sel_fly)]
                st.session_state.headsail_id = chosen_fly.sail_id
                headsail_geom = chosen_fly
                luff = getattr(chosen_fly, "JIBLUFF", None) or getattr(chosen_fly, "SLU", 0.0)
                st.caption(f"Area: {chosen_fly.sail_area:.1f} m²  ·  Luff: {luff:.2f} m")
        else:
            st.caption("No DXT loaded — using default IMOCA 60 dimensions (J1.5 / A6.5).")

    with col_app2:
        st.subheader("Appendages")
        use_foil    = st.checkbox("Deploy leeward foil", value=True)
        foil_cant   = st.slider("Foil rake angle [°]", 0, 60, 40)
        keel_angle_deg = st.slider("Keel cant angle [°]", -45, 45, 0)
        st.caption("The translated Simulink path currently keeps the rudder branch at zero, matching the source model wiring.")

    with col_time:
        st.subheader("Integration")
        t_end = st.slider("Simulation time [s]", 10, 120, 40)
        dt    = st.selectbox("Time step [s]", [0.02, 0.05, 0.10], index=1)
        sim_mass_kg = st.number_input(
            "Simulation mass [kg]",
            value=float(st.session_state.mass_kg),
            step=100.0,
        )

        st.markdown("**Radiation / Cummins source**")
        rad_options = ["Original Simulink (NEMOH)"]
        if st.session_state.solver_done:
            rad_options.append("Panel Solver (strip theory / BEM)")
        rad_source = st.radio(
            "Hydrodynamic coefficients",
            rad_options,
            index=0,
            help="Choose which B(ω), A(ω) and A_inf feed the Cummins radiation equation. "
                 "Run the Panel Solver first to unlock the second option.",
        )
        if rad_source.startswith("Panel") and st.session_state.solver_done:
            _s = st.session_state.solver
            radiation_data = dict(
                B_omega=_s.B_omega,
                A_omega=_s.A_omega,
                A_inf=_s.A_inf,
                omega=_s.omega,
            )
            st.caption(f"Using panel-solver coefficients ({len(_s.omega)} freq. points).")
        else:
            radiation_data = None
            if not st.session_state.solver_done:
                st.caption("Run the Panel Solver (Tab 1) to enable its coefficients here.")

    with st.expander("Initial Conditions", expanded=False):
        col_eta0, col_nu0 = st.columns(2)

        with col_eta0:
            st.markdown("**Position / attitude**")
            x0 = st.number_input("x0 [m]", value=0.0, step=0.1)
            y0 = st.number_input("y0 [m]", value=0.0, step=0.1)
            z0 = st.number_input("z0 [m]", value=0.0, step=0.05)
            heel0_deg = st.number_input("Heel φ0 [deg]", value=0.0, step=0.5)
            trim0_deg = st.number_input("Trim θ0 [deg]", value=0.0, step=0.5)
            yaw0_deg = st.number_input("Yaw ψ0 [deg]", value=0.0, step=0.5)

        with col_nu0:
            st.markdown("**Velocity / rates**")
            start_bsp_kn = st.number_input("Start BSP u0 [kn]", value=0.0, step=0.1)
            sway0_kn = st.number_input("Sway v0 [kn]", value=0.0, step=0.1)
            heave0 = st.number_input("Heave w0 [m/s]", value=0.0, step=0.05)
            roll_rate0 = st.number_input("p0 [deg/s]", value=0.0, step=0.5)
            pitch_rate0 = st.number_input("q0 [deg/s]", value=0.0, step=0.5)
            yaw_rate0 = st.number_input("r0 [deg/s]", value=0.0, step=0.5)

    with st.expander("Model Inputs", expanded=False):
        col_model_l, col_model_r = st.columns(2)

        with col_model_l:
            hull_path = st.text_input("Hull STL path", value=sim_default_hull)
            use_manual_waterline = st.checkbox("Override waterline z", value=False)
            sim_waterline_z = None
            if use_manual_waterline:
                sim_waterline_z = st.number_input("Waterline z [m]", value=0.0, step=0.01)

        with col_model_r:
            st.markdown("**Wave note**")
            st.caption(
                "The active runtime now takes wave mode, wave ramp, wave heading, "
                "and wave wind speed from the UI instead of forcing calm water."
            )
            st.markdown("**Initial-state note**")
            st.caption(
                "Use `Start BSP u0` for the initial boat speed. All pose and rate "
                "inputs are passed directly into the translated Simulink runtime."
            )

    st.markdown("---")
    col_run, col_info = st.columns([1, 3])

    with col_info:
        st.info("The Simulation tab runs only on the translated Simulink model. Wind, waves, initial state, hull path, mass, and appendage inputs are taken from the UI.")

    with col_run:
        run_btn = st.button("▶️ Run Simulation", type="primary")

    if run_btn:
        if not os.path.exists(hull_path):
            st.error(f"Hull STL not found at: {hull_path}")
            st.stop()

        eta0 = np.array([
            float(x0),
            float(y0),
            float(z0),
            np.deg2rad(float(heel0_deg)),
            np.deg2rad(float(trim0_deg)),
            np.deg2rad(float(yaw0_deg)),
        ], dtype=float)
        nu0 = np.array([
            float(start_bsp_kn) / 1.943844,
            float(sway0_kn) / 1.943844,
            float(heave0),
            np.deg2rad(float(roll_rate0)),
            np.deg2rad(float(pitch_rate0)),
            np.deg2rad(float(yaw_rate0)),
        ], dtype=float)

        with st.spinner(f"Integrating {t_end}s with dt={dt}s …"):
            try:
                t0 = time.time()
                sim_outputs = run_simulation(
                    twa_deg       = float(TWA),
                    tws_kn        = float(TWS),
                    reef          = float(Reef),
                    flat          = float(Flat),
                    c0_enabled    = bool(C0_enabled),
                    use_foil      = bool(use_foil),
                    rake_foil_deg = float(foil_cant),
                    keel_angle_deg= float(keel_angle_deg),
                    t_end         = float(t_end),
                    dt            = float(dt),
                    hull_file     = hull_path,
                    mass          = float(sim_mass_kg),
                    waterline_z   = sim_waterline_z,
                    eta0          = eta0,
                    nu0           = nu0,
                    wave_mode     = wave_mode,
                    wave_ramp     = float(wave_ramp),
                    wave_angle_deg= float(wave_angle_deg),
                    wave_wind_speed_kn = float(wave_wind_speed_kn),
                    mainsail_geom = mainsail_geom,
                    jib_geom      = jib_geom,
                    headsail_geom = headsail_geom,
                    radiation_data= radiation_data,
                )
                elapsed = time.time() - t0
                st.session_state.t_vals  = sim_outputs.t_vals
                st.session_state.y_vals  = sim_outputs.y_vals
                st.session_state.sim_outputs = sim_outputs
                st.session_state.sim_done = True
                st.session_state.mass_kg = float(sim_mass_kg)
                st.success(f"Simulation complete in {elapsed:.1f}s. Waterline z = {sim_outputs.waterline_z:.3f} m")
            except Exception as e:
                st.error(f"Simulation error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Simulation Results")

    if not st.session_state.sim_done:
        st.info("Run a simulation in the Simulation tab first.")
    else:
        t_vals = st.session_state.t_vals
        y_vals = st.session_state.y_vals

        # --- Velocity ---
        st.subheader("Velocities (body frame)")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

        ax = axes[0]
        ax.plot(t_vals, y_vals[:, 6] * 1.943844, label='u — surge [kn]')
        ax.plot(t_vals, y_vals[:, 7] * 1.943844, label='v — sway [kn]')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [kn]')
        ax.set_title('Surge & sway speed'); ax.legend(); ax.grid()

        ax = axes[1]
        ax.plot(t_vals, y_vals[:, 8], label='w — heave [m/s]', color='purple')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('w [m/s]')
        ax.set_title('Heave velocity'); ax.legend(); ax.grid()

        st.pyplot(fig); plt.close(fig)

        # --- Position ---
        st.subheader("World-frame position")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

        ax = axes[0]
        ax.plot(t_vals, y_vals[:, 0], label='x [m]')
        ax.plot(t_vals, y_vals[:, 1], label='y [m]')
        ax.set_xlabel('Time [s]'); ax.legend(); ax.grid()
        ax.set_title('Surge & sway position')

        ax = axes[1]
        ax.plot(t_vals, y_vals[:, 2], label='z [m]', color='teal')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, label='waterline')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('z [m]')
        ax.set_title('Heave position (positive up)'); ax.legend(); ax.grid()

        st.pyplot(fig); plt.close(fig)

        # --- Angles ---
        st.subheader("Orientation (Euler angles)")
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(t_vals, np.rad2deg(y_vals[:, 3]), label='φ — heel [°]')
        ax.plot(t_vals, np.rad2deg(y_vals[:, 4]), label='θ — trim [°]')
        ax.plot(t_vals, np.rad2deg(y_vals[:, 5]), label='ψ — yaw [°]')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Angle [°]')
        ax.set_title('Euler angles'); ax.legend(); ax.grid()
        st.pyplot(fig); plt.close(fig)

        # --- Summary metrics ---
        st.subheader("Performance metrics")
        final_u = y_vals[-1, 6] * 1.943844
        max_heel = np.max(np.abs(np.rad2deg(y_vals[:, 3])))
        max_z    = np.max(np.abs(y_vals[:, 2]))

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Final boat speed", f"{final_u:.2f} kn")
        mc2.metric("Max heel angle",   f"{max_heel:.1f}°")
        mc3.metric("Max heave amp.",   f"{max_z:.3f} m")

        # --- Download CSV ---
        import io, pandas as pd
        cols = ['x','y','z','phi','theta','psi','u','v','w','p','q','r']
        df = pd.DataFrame(y_vals, columns=cols)
        df.insert(0, 't', t_vals)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download results (CSV)",
            data=csv,
            file_name='dvpp_results.csv',
            mime='text/csv',
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Hydrodynamic Coefficients")

    if not st.session_state.solver_done:
        st.info("Run the Panel Solver in Tab 1 to inspect hydrodynamic coefficients.")
    else:
        solver = st.session_state.solver
        omega  = solver.omega
        B      = solver.B_omega
        A      = solver.A_omega

        dof_labels = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']

        st.subheader("Radiation damping B(ω) — diagonal terms")
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, (ax, lbl) in enumerate(zip(axes.flat, dof_labels)):
            ax.plot(omega, B[i, i, :])
            ax.set_title(f'B_{i+1}{i+1} — {lbl}')
            ax.set_xlabel('ω [rad/s]')
            ax.set_ylabel('B [N·s/m or N·m·s/rad]')
            ax.grid()
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.subheader("Added mass A(ω) — diagonal terms")
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i, (ax, lbl) in enumerate(zip(axes.flat, dof_labels)):
            ax.plot(omega, A[i, i, :])
            ax.axhline(solver.A_inf[i, i], color='r', linestyle='--',
                       linewidth=0.8, label='A_inf')
            ax.set_title(f'A_{i+1}{i+1} — {lbl}')
            ax.set_xlabel('ω [rad/s]')
            ax.set_ylabel('A [kg or kg·m²]')
            ax.legend(); ax.grid()
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.subheader("Hydrostatic restoring matrix C")
        import pandas as pd
        C_df = pd.DataFrame(
            solver.C,
            index=dof_labels, columns=dof_labels
        )
        st.dataframe(C_df.style.format("{:.2f}"))

        st.subheader("Infinite-frequency added mass A_inf")
        Ai_df = pd.DataFrame(
            solver.A_inf,
            index=dof_labels, columns=dof_labels
        )
        st.dataframe(Ai_df.style.format("{:.1f}"))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEAVE DROP TEST
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Heave Drop Test (Cummins 1-DOF)")
    st.markdown(
        """
        Simulates dropping the vessel from a height **h** above the still-water
        surface and integrating the **Cummins equation**:

        **(M + A₃₃_inf) ẍ + ∫ K(t−s) ẋ(s) ds + B_visc·ẋ + C₃₃ x = 0**

        **Why does radiation theory give almost no heave damping for a ship?**

        For a slender ship the heave natural frequency ω₀ is typically **3–5 rad/s**.
        At those frequencies the Kochin integral `∫b(x)e^{ik₀x}dx` is highly
        oscillatory (k₀L ≫ 1) → **B₃₃(ω₀) ≈ 0**.  This is correct physics:
        radiation damping alone is genuinely tiny for slender hulls at high ω.

        Real ships are damped primarily by **viscous effects** (skin friction,
        keel drag, bilge keels) — not modelled by potential theory.  Use the
        **B_visc** slider below to add a representative viscous term.

        > Rule of thumb: B_visc ≈ 5–20% of B_critical = 2√(C₃₃·M_tot)

        **Run the Panel Solver (Tab 1) first.**
        """
    )

    if not st.session_state.solver_done:
        st.info("Run the Panel Solver in Tab 1 first.")
    else:
        solver_dt = st.session_state.solver
        omega_dt  = solver_dt.omega
        B33_dt    = solver_dt.B_omega[2, 2]   # (N_omega,)
        A33_dt    = solver_dt.A_omega[2, 2]   # (N_omega,)
        A33inf_dt = float(solver_dt.A_inf[2, 2])
        C33_dt    = float(solver_dt.C[2, 2])

        col_l5, col_r5 = st.columns([1, 2])

        with col_l5:
            st.subheader("Drop settings")

            mass_dt = st.number_input(
                "Vessel mass [kg]",
                value=float(st.session_state.mass_kg),
                step=100.0,
                help="Total vessel mass.  Should match the mass used in the panel solver."
            )
            h_drop_dt = st.number_input(
                "Drop height above still water [m]",
                value=1.0, min_value=0.01, step=0.1
            )
            t_end_dt = st.number_input(
                "Simulation duration [s]", value=60.0, step=5.0, min_value=5.0
            )
            dt_dt = st.selectbox("Time step [s]", [0.01, 0.02, 0.05], index=1)

            st.markdown("---")
            M_tot_dt = mass_dt + A33inf_dt
            if C33_dt > 0 and M_tot_dt > 0:
                omega0_dt = np.sqrt(C33_dt / M_tot_dt)
                B33_at_w0 = float(np.interp(omega0_dt, omega_dt, B33_dt))
                B_crit_dt = 2.0 * np.sqrt(C33_dt * M_tot_dt)
                zeta_rad  = B33_at_w0 / B_crit_dt
                st.metric("Natural frequency ω₀", f"{omega0_dt:.3f} rad/s")
                st.metric("Period T₀", f"{2*np.pi/omega0_dt:.2f} s")
                st.metric("B₃₃(ω₀) — strip theory", f"{B33_at_w0:,.0f} N·s/m")
                # NEMOH reference at same ω₀
                B33_nem_at_w0_disp = float(np.interp(omega0_dt, _NEMOH_OMEGA, _NEMOH_B33))
                st.metric("B₃₃(ω₀) — NEMOH 3D BEM", f"{B33_nem_at_w0_disp:,.0f} N·s/m")
                st.metric("B_critical", f"{B_crit_dt:,.0f} N·s/m")
                st.metric("ζ_strip (radiation)", f"{zeta_rad:.4f}")
                zeta_nem_disp = B33_nem_at_w0_disp / B_crit_dt
                st.metric("ζ_NEMOH (radiation)", f"{zeta_nem_disp:.4f}")

            st.markdown("**Viscous damping supplement**")
            B_visc_pct = st.slider(
                "B_visc as % of B_critical",
                min_value=0, max_value=50, value=10, step=1,
                help="Adds B_visc·ẋ to the equation — represents skin friction, "
                     "keel drag, bilge keels.  10% is a typical starting point."
            )
            if C33_dt > 0 and M_tot_dt > 0:
                B_visc_dt  = (B_visc_pct / 100.0) * B_crit_dt
                zeta_total = (B33_at_w0 + B_visc_dt) / B_crit_dt
                st.caption(f"B_visc = {B_visc_dt:,.0f} N·s/m  →  ζ_total = {zeta_total:.4f}")
            else:
                B_visc_dt  = 0.0
                zeta_total = 0.0

            if st.button("🏊 Run Drop Test", type="primary"):
                v0_dt = np.sqrt(2.0 * 9.81 * h_drop_dt)
                with st.spinner("Computing retardation kernel and integrating …"):
                    try:
                        # ── Solver (strip theory or Capytaine) ──────────────
                        tau_K_dt, K_dt = _retardation_kernel(B33_dt, omega_dt)
                        t_sim_dt, z_sim_dt = _simulate_cummins(
                            M_tot_dt, C33_dt, tau_K_dt, K_dt, v0_dt,
                            t_end=float(t_end_dt), dt=float(dt_dt),
                            B_visc=float(B_visc_dt),
                        )
                        t_ana_dt = np.linspace(0, float(t_end_dt), 3000)
                        z_ana_dt = _analytical_drop(
                            M_tot_dt, B33_at_w0 + B_visc_dt, C33_dt,
                            v0_dt, t_ana_dt
                        )

                        # ── NEMOH reference (hardcoded 3D BEM for IMOCA 60) ─
                        # Interpolate NEMOH B33 onto the solver omega grid
                        B33_nem_ref = np.interp(omega_dt, _NEMOH_OMEGA, _NEMOH_B33,
                                                left=0.0, right=0.0)
                        A33inf_nem  = float(_NEMOH_A_INF[2, 2])
                        M_tot_nem   = float(mass_dt) + A33inf_nem
                        B33_nem_at_w0 = float(np.interp(omega0_dt, _NEMOH_OMEGA, _NEMOH_B33))
                        B_crit_nem  = 2.0 * np.sqrt(C33_dt * M_tot_nem)
                        zeta_nem    = B33_nem_at_w0 / B_crit_nem
                        tau_K_nem, K_nem = _retardation_kernel(B33_nem_ref, omega_dt)
                        t_sim_nem, z_sim_nem = _simulate_cummins(
                            M_tot_nem, C33_dt, tau_K_nem, K_nem, v0_dt,
                            t_end=float(t_end_dt), dt=float(dt_dt),
                        )

                        st.session_state.drop_results = {
                            'omega'       : omega_dt,
                            'B33'         : B33_dt,
                            'A33'         : A33_dt,
                            'A33inf'      : A33inf_dt,
                            'C33'         : C33_dt,
                            'tau_K'       : tau_K_dt,
                            'K'           : K_dt,
                            't_sim'       : t_sim_dt,
                            'z_sim'       : z_sim_dt,
                            't_ana'       : t_ana_dt,
                            'z_ana'       : z_ana_dt,
                            'omega0'      : omega0_dt,
                            'zeta_rad'    : zeta_rad,
                            'zeta_total'  : zeta_total,
                            'B_visc'      : B_visc_dt,
                            'h_drop'      : h_drop_dt,
                            'v0'          : v0_dt,
                            # NEMOH reference curves
                            'B33_nem'     : B33_nem_ref,
                            'tau_K_nem'   : tau_K_nem,
                            'K_nem'       : K_nem,
                            't_sim_nem'   : t_sim_nem,
                            'z_sim_nem'   : z_sim_nem,
                            'zeta_nem'    : zeta_nem,
                            'B33_nem_at_w0': B33_nem_at_w0,
                        }
                        st.session_state.drop_done = True
                        st.success("Drop test complete.")
                    except Exception as e:
                        st.error(f"Drop test error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        with col_r5:
            if st.session_state.drop_done:
                dr = st.session_state.drop_results
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # [0,0] B₃₃(ω)
                ax = axes[0, 0]
                ax.plot(dr['omega'], dr['B33'] / 1e3, 'r-', lw=2.0,
                        label='Panel solver (strip theory)')
                ax.plot(dr['omega'], dr['B33_nem'] / 1e3, 'g--', lw=2.0,
                        label=f"NEMOH 3D BEM  ζ={dr['zeta_nem']:.4f}")
                ax.axvline(dr['omega0'], ls=':', color='steelblue',
                           label=f"ω₀ = {dr['omega0']:.2f} rad/s")
                ax.set_xlabel('ω [rad/s]')
                ax.set_ylabel('B₃₃ [kN·s/m]')
                ax.set_title('Heave radiation damping B₃₃(ω)')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

                # [0,1] A₃₃(ω)
                ax = axes[0, 1]
                ax.plot(dr['omega'], dr['A33'] / 1e3, 'r-', lw=2.0,
                        label='Panel solver A₃₃')
                ax.axhline(dr['A33inf'] / 1e3, ls=':', color='k', lw=1.5,
                           label=f"A₃₃_inf (solver) = {dr['A33inf']/1e3:.1f} t")
                ax.axhline(_NEMOH_A_INF[2, 2] / 1e3, ls=':', color='g', lw=1.5,
                           label=f"A₃₃_inf (NEMOH) = {_NEMOH_A_INF[2,2]/1e3:.1f} t")
                ax.axvline(dr['omega0'], ls=':', color='steelblue',
                           label=f"ω₀ = {dr['omega0']:.2f} rad/s")
                ax.set_xlabel('ω [rad/s]')
                ax.set_ylabel('A₃₃ [t]')
                ax.set_title('Heave added mass A₃₃(ω)')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

                # [1,0] Retardation kernel
                ax = axes[1, 0]
                ax.plot(dr['tau_K'], dr['K'] / 1e3, 'r-', lw=1.8,
                        label='Strip theory K(τ)')
                ax.plot(dr['tau_K_nem'], dr['K_nem'] / 1e3, 'g--', lw=1.8,
                        label='NEMOH K(τ)')
                ax.axhline(0, color='k', lw=0.6)
                ax.set_xlabel('τ [s]')
                ax.set_ylabel('K(τ) [kN/m]')
                ax.set_title('Retardation kernel K(τ)  — Ogilvie transform of B₃₃(ω)')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

                # [1,1] Drop test
                ax = axes[1, 1]
                visc_label = (f" + B_visc={dr['B_visc']/1e3:.1f} kN·s/m"
                              if dr['B_visc'] > 0 else " (radiation only)")
                ax.plot(dr['t_ana'], dr['z_ana'] * 100, 'r-', lw=2.5,
                        label=f"Analytical (strip)  ζ = {dr['zeta_total']:.4f}")
                ax.plot(dr['t_sim'], dr['z_sim'] * 100, 'r--', lw=2.0,
                        label=f'Cummins (strip{visc_label})')
                ax.plot(dr['t_sim_nem'], dr['z_sim_nem'] * 100, 'g-', lw=2.0,
                        label=f"Cummins (NEMOH 3D BEM)  ζ = {dr['zeta_nem']:.4f}")
                ax.axhline(0, color='gray', lw=0.5)
                ax.set_xlabel('t [s]')
                ax.set_ylabel('z [cm]')
                ax.set_title(
                    f"Heave drop  h = {dr['h_drop']} m  "
                    f"(v₀ = {dr['v0']:.2f} m/s)"
                )
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

                fig.suptitle(
                    f"Hull heave drop test  |  ω₀ = {dr['omega0']:.3f} rad/s"
                    f"  |  ζ_strip = {dr['zeta_total']:.4f}"
                    f"  |  ζ_NEMOH = {dr['zeta_nem']:.4f}",
                    fontsize=11, fontweight='bold'
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Configure the drop parameters and click **Run Drop Test**.")
