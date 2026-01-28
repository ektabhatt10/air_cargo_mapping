from Optimizer import run_optimization
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import ast
import streamlit as st
st.set_page_config(layout="wide")

from fpdf import FPDF

def generate_full_pdf(assignments_df, summary_df, table):
    from reportlab.platypus import SimpleDocTemplate, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate("full_report.pdf", pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Spacer(1, 12))
    from reportlab.platypus import Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    elements.append(Paragraph("📦 Parcel Assignment Table", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(table)
    doc.build(elements)

# --- Step 1: Upload Files ---
st.subheader("📁 Upload Input Files")
parcel_file = st.file_uploader("Upload Parcel File (.xlsx)", type="xlsx")
container_file = st.file_uploader("Upload Container File (.xlsx)", type="xlsx")

if parcel_file:
    parcels_df = pd.read_excel(parcel_file, sheet_name="Parcels")
    st.write("📦 Parcels Preview")
    st.dataframe(parcels_df)
    st.session_state["total_parcels_in_file"] = len(parcels_df)

if container_file:
    containers_df = pd.read_excel(container_file, sheet_name="Containers")
    containers_df = containers_df[containers_df["Qty"] > 0]
    st.write("🚚 Containers Preview")
    st.dataframe(containers_df)

    # Build a lookup dictionary for container specs
    container_specs_dict = {}
    for _, row in containers_df.iterrows():
        container_specs_dict[row["ContainerID"]] = {
            "L (cm)": row["L (cm)"],
            "W (cm)": row["W (cm)"],
            "H (cm)": row["H (cm)"],
            "MaxPayload_kg": row["Maximum Payload (kg)"]
        }

if not parcel_file or not container_file:
    st.info("Please upload both files to continue.")
    st.stop()

# --- Step 2: User Inputs ---
st.header("Configuration Options")
col1, col2, col3 = st.columns(3)
with col1:
    clearance_L = st.number_input("Clearance Length (cm)", min_value=0, value=0)
with col2:
    clearance_W = st.number_input("Clearance Width (cm)", min_value=0, value=0)
with col3:
    clearance_H = st.number_input("Clearance Height (cm)", min_value=0, value=0)

col4, col5 = st.columns(2)
with col4:
    inter_box_clearance_L = st.number_input("Inter-box Clearance (Length, cm)", min_value=0.0, value=0.0)
with col5:
    inter_row_clearance_W = st.number_input("Inter-row Clearance (Width, cm)", min_value=0.0, value=0.0)

sharing_allowed = st.checkbox("Allow sharing of container space", value=True)

# --- Step 3: Run Optimization ---
run_clicked = st.button("🚀 Run Optimization")
if run_clicked:
    settings = {
        "wall_clearance_L_cm": clearance_L,
        "wall_clearance_W_cm": clearance_W,
        "wall_clearance_H_cm": clearance_H,
        "inter_box_clearance_L_cm": inter_box_clearance_L,
        "inter_row_clearance_W_cm": inter_row_clearance_W,
        "max_containers": 0,
        "priority_rule": "lower_number_higher_priority"
    }
    containers_df = containers_df[containers_df["Qty"] > 0]
    assignments_df = run_optimization(parcels_df, containers_df, settings, sharing_allowed)

    def safe_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return val
        return val

    assignments_df["Position"] = assignments_df["Position"].apply(safe_eval)
    assignments_df["Orientation"] = assignments_df["Orientation"].apply(safe_eval)

    assignments_df["Volume_m3"] = (
        assignments_df["Length_cm"] * assignments_df["Width_cm"] * assignments_df["Height_cm"]
    ) / 1e6

    assignments_df.drop(columns=["Volume_cm3", "VolumetricWeight_kg", "ChargeableWeight_kg"], inplace=True, errors="ignore")

    # ✅ Save for later use
    st.session_state["assignments_df"] = assignments_df
    st.session_state["container_specs_dict"] = container_specs_dict

# --- View Toggle and Visualization ---
if "assignments_df" in st.session_state and "container_specs_dict" in st.session_state:
    st.subheader("🔍 Choose View")
    view_option = st.radio("Select a view:", ["None", "📊 Container Utilization", "🧭 3D Mapping"], horizontal=True)

    assignments_df = st.session_state["assignments_df"]
    container_specs_dict = st.session_state["container_specs_dict"]

    # Show assignment table and summary again
    st.subheader("📋 Parcel Assignments")
    st.dataframe(assignments_df)

    # --- Global summary (based on assignments_df) ---
    total_parcels_in_inventory = st.session_state.get("total_parcels_in_file", len(assignments_df))
    total_parcels_in_valid_lanes = len(assignments_df)
    placed_df = assignments_df[assignments_df["ContainerID"].notnull()]
    containers_used = placed_df["ContainerID"].nunique()

    st.subheader("📦 Global Summary")
    colA, colB, colC, colD= st.columns(4)
    colA.metric("Total Parcels in File", total_parcels_in_inventory)
    colB.metric("Parcels in Valid Trade Lanes", total_parcels_in_valid_lanes)
    colC.metric("Placed Parcels", len(placed_df))
    colD.metric("Containers Used", containers_used)

    # --- Per-trade-lane summary ---
    st.subheader("🛫 Trade Lane Summary")

    lane_summary = (
        assignments_df
        .groupby(["Origin Hub Airport", "Destination Hub Airport"])
        .agg(
            TotalParcels=("ParcelID", "count"),
            PlacedParcels=("ContainerID", lambda x: x.notna().sum()),
            ContainersUsed=("ContainerID", lambda x: x.dropna().nunique())
        )
        .reset_index()
    )

    st.dataframe(lane_summary)


    # 📦 Container Summary Table
    st.subheader("🚚 Container Summary")
    summary_rows = []
    for container_id in assignments_df["ContainerID"].dropna().unique():
        container_df = assignments_df[assignments_df["ContainerID"] == container_id]
        specs = container_specs_dict.get(container_id)
        if specs is None:
            continue

        total_volume_m3 = (specs["L (cm)"] * specs["W (cm)"] * specs["H (cm)"]) / 1e6
        used_volume_m3 = container_df["Volume_m3"].sum()
        volume_util_pct = used_volume_m3 / total_volume_m3 * 100

        max_payload_kg = specs["MaxPayload_kg"]
        used_weight_kg = container_df["Weight_kg"].sum()
        weight_util_pct = (used_weight_kg / max_payload_kg * 100) if max_payload_kg > 0 else 0

        summary_rows.append({
            "ContainerID": container_id,
            "UsedVolume_m3": round(used_volume_m3, 3),
            "TotalVolume_m3": round(total_volume_m3, 3),
            "VolumeUtilization_%": round(volume_util_pct, 1),
            "UsedWeight_kg": round(used_weight_kg, 2),
            "MaxPayload_kg": round(max_payload_kg, 2),
            "WeightUtilization_%": round(weight_util_pct, 1)
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    # --- Container Utilization ---
    if view_option == "📊 Container Utilization":
        for container_id in assignments_df["ContainerID"].unique():
            container_df = assignments_df[assignments_df["ContainerID"] == container_id]
            specs = container_specs_dict.get(container_id)
            if specs is None:
                st.warning(f"Unknown container: {container_id}")
                continue

            container_volume_cm3 = specs["L (cm)"] * specs["W (cm)"] * specs["H (cm)"]
            used_volume_cm3 = sum([l * w * h for l, w, h in container_df["Orientation"]])
            used_weight_kg = container_df["Weight_kg"].sum()
            max_payload_kg = specs["MaxPayload_kg"]

            volume_util_pct = used_volume_cm3 / container_volume_cm3 * 100
            weight_util_pct = used_weight_kg / max_payload_kg * 100

            fig_bar, ax = plt.subplots(figsize=(6, 2))
            labels = ['Volume Utilization', 'Weight Utilization']
            values = [min(volume_util_pct, 100), min(weight_util_pct, 100)]
            colors = ['skyblue', 'lightgreen']
            bars = ax.barh(labels, values, color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Utilization (%)')
            ax.set_title(f'{container_id} Utilization')
            for bar, actual in zip(bars, [volume_util_pct, weight_util_pct]):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{actual:.1f}%', va='center')
            st.pyplot(fig_bar)

            if used_weight_kg > max_payload_kg:
                st.error(f"⚠️ {container_id} is overloaded: {used_weight_kg:.1f} kg > {max_payload_kg} kg")

    # --- 3D Mapping ---
    if view_option == "🧭 3D Mapping":
        brands = assignments_df["Brand"].unique()
        brand_color_map = {
            brand: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, brand in enumerate(brands)
        }

        for container_id in assignments_df["ContainerID"].dropna().unique():
            st.subheader(f"📦 {container_id} Layout")

            container_df = assignments_df[assignments_df["ContainerID"] == container_id]
            specs = container_specs_dict.get(container_id)
            if specs is None:
                st.warning(f"Unknown container: {container_id}")
                continue

            fig3d = go.Figure() 
            fig3d.add_trace(go.Scatter3d(
                x=[0, specs["L (cm)"]], y=[0, specs["W (cm)"]], z=[0, specs["H (cm)"]],
                mode='markers',
                marker=dict(size=0.1),
                name=f'{container_id} Container'
            ))

            for _, row in container_df.iterrows():
                x0, y0, z0 = row["Position"]
                l, w, h = row["Orientation"]
                parcel_id = row["ParcelID"]
                weight = row["Weight_kg"]
                brand = row.get("Brand", "Unknown")
                color = brand_color_map.get(brand, "gray")

                x = [x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0]
                y = [y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w]
                z = [z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h]

                fig3d.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    color=color,
                    opacity=0.6,
                    name=parcel_id,
                    hovertext=f"{parcel_id}<br>{l}×{w}×{h} cm<br>{weight} kg<br>{brand}<br>{container_id}",
                    hoverinfo="text"
                ))

            fig3d.update_layout(
                scene=dict(
                    xaxis_title='Length (cm)',
                    yaxis_title='Width (cm)',
                    zaxis_title='Height (cm)',
                    aspectmode='manual',
                    aspectratio=dict(x=2, y=1.5, z=1)
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig3d, use_container_width=True)

    # --- Export Results ---
    with open("C:/Users/ektab/container_packing_results.xlsx", "rb") as f:
        st.download_button(
            label="📥 Download Results as Excel",
            data=f.read(),
            file_name="container_packing_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    table_data = assignments_df[[
    "Origin Hub Airport","Destination Hub Airport","ParcelID", "ContainerID", "Container Brand",
    "Length_cm", "Width_cm", "Height_cm",
    "Weight_kg", "Position", "Orientation"
    ]]

    table_data["Position"] = table_data["Position"].apply(lambda x: f"{x[0]},{x[1]},{x[2]}" if x else "")
    table_data["Orientation"] = table_data["Orientation"].apply(lambda x: f"{x[0]},{x[1]},{x[2]}" if x else "")

    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    # Convert to list of lists
    header = ["ParcelID", "ContainerID", "Brand", "L*W*H (cm)", "Weight (kg)", "Position", "Orientation"]
    rows = []

    for _, row in table_data.iterrows():
        dims = f"{row['Length_cm']}*{row['Width_cm']}*{row['Height_cm']}"
        rows.append([
            row["ParcelID"], row["ContainerID"], row["Container Brand"],
            dims, row["Weight_kg"], row["Position"], row["Orientation"]
        ])

    table = Table([header] + rows)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))

    generate_full_pdf(assignments_df, summary_df, table)
    with open("full_report.pdf", "rb") as f:
        st.download_button(
            label="📥 Download Full Report as PDF",
            data=f.read(),
            file_name="container_optimization_report.pdf",
            mime="application/pdf"
        )    

if st.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()
