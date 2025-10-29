def run_optimization(parcels_df, containers_df, settings, sharing_allowed):
    import os
    os.getcwd()

    import pandas as pd
    import numpy as np

    # Preview the data
    print("Parcels:")
    import streamlit as st
    print(parcels_df.head())

    print("\nContainers:")
    print(containers_df.head())

    # -------------------------
    # CLEARANCE (units: centimetres)
    # -------------------------
    # Build a simple clearance dict from your settings (put this in the same cell)
    clearance = {
        "wall_L": float(settings.get("wall_clearance_L_cm", 0)),
        "wall_W": float(settings.get("wall_clearance_W_cm", 0)),
        "wall_H": float(settings.get("wall_clearance_H_cm", 0)),
        "inter_L": float(settings.get("inter_box_clearance_L_cm", 0)),
        "inter_W": float(settings.get("inter_row_clearance_W_cm", 0))
    }

    def fits_in_container(parcel_dim, container_dim, clearance):
        """
        Check if a parcel (in one orientation) fits dimensionally into a container,
        after subtracting wall clearances on both sides.

        parcel_dim: tuple/list (Lp, Wp, Hp) in cm
        container_dim: tuple/list (Lc, Wc, Hc) in cm
        clearance: dict with keys "wall_L","wall_W","wall_H" (all in cm)

        Returns: True/False
        """
        # --- Input validation ---
        if not (isinstance(parcel_dim, (list, tuple)) and len(parcel_dim) == 3):
            raise ValueError("parcel_dim must be a tuple/list of three numbers (L,W,H).")
        if not (isinstance(container_dim, (list, tuple)) and len(container_dim) == 3):
            raise ValueError("container_dim must be a tuple/list of three numbers (L,W,H).")

        try:
            Lp, Wp, Hp = map(float, parcel_dim)
            Lc, Wc, Hc = map(float, container_dim)
        except Exception:
            raise ValueError("parcel_dim and container_dim must contain numeric values.")

        # --- get clearance values (fall back to 0 if missing) ---
        wall_L = float(clearance.get("wall_L", clearance.get("wall_clearance_L_cm", 0)))
        wall_W = float(clearance.get("wall_W", clearance.get("wall_clearance_W_cm", 0)))
        wall_H = float(clearance.get("wall_H", clearance.get("wall_clearance_H_cm", 0)))

        # --- effective usable container dims after subtracting clearances on both sides ---
        Lc_eff = Lc - 2.0 * wall_L
        Wc_eff = Wc - 2.0 * wall_W
        Hc_eff = Hc - wall_H

        # Defensive: if effective dims are non-positive, it can't fit
        if Lc_eff <= 0 or Wc_eff <= 0 or Hc_eff <= 0:
            return False

        # small epsilon to avoid floating point edge issues
        eps = 1e-9
        return (Lp <= Lc_eff + eps) and (Wp <= Wc_eff + eps) and (Hp <= Hc_eff + eps)

    # Use the columns already in cm
    try:
        parcels_df["Volume_cm3"] = (
            parcels_df["L (cm)"] * parcels_df["W (cm)"] * parcels_df["H (cm)"]
        )
        print("Parcel volume computed successfully")
    except Exception as e:
        print("Volume computation failed:", e)

    try:
        parcels_df["VolumetricWeight_kg"] = (
        parcels_df["L (cm)"] * 
        parcels_df["W (cm)"] * 
        parcels_df["H (cm)"]
    ) / 6000
        print("Parcel Volumetric weight computed successfully")
    except Exception as e:
        print("Volumetric weight computation failed:",e)

    try:
        containers_df["Volume_cm3"] = (
            containers_df["L (cm)"] * containers_df["W (cm)"] * containers_df["H (cm)"]
        )
        print("Container volume computed successfully")
    except Exception as e:
        print("Volume computation failed:", e)

    # --- Debug cargo and container stats ---
    print("📦 Containers summary:")
    print(containers_df[["ContainerID", "Volume_cm3", "Maximum Payload (kg)", "Tare Weight (kg)"]])

    print("\n📦 Cargo summary:")
    print("Total Volume (m³):", round(parcels_df["Volume_cm3"].sum() / 1e6, 3))
    print("Total Weight (kg):", round(parcels_df["Weight (kg)"].sum(), 2))

    # Preview parcels
    print("Parcels preview:")
    print(parcels_df.head())

    # Preview containers
    print("\nContainers preview:")
    print(containers_df.head())

    # Check if 'Priority' column exists
    if "Priority" not in parcels_df.columns:
        raise ValueError("Parcels sheet must have a 'Priority' column")

    # Sort parcels by:
    # 1) Priority (lower number = higher priority)
    # 2) Volume (largest first within same priority)
    parcels_df_sorted = parcels_df.sort_values(
        by=["Priority", "Volume_cm3"],
        ascending=[True, False]
    ).reset_index(drop=True) 

    print("Parcels sorted by priority and volume:")
    print(parcels_df_sorted.head(10))  # show first 10 for sanity check

    # If "Turnable" column exists, convert to boolean
    if "Turnable" in parcels_df_sorted.columns:
        parcels_df_sorted["Turnable"] = parcels_df_sorted["Turnable"].apply(lambda x: str(x).strip().lower() in ["true", "1", "yes"])
    else:
        # If no column, assume all turnable
        parcels_df_sorted["Turnable"] = True

    def get_orientations(length, width, height, turnable=True):
        if not turnable:
            # Only upright base, but can rotate on ground plane
            return [
                (length, width, height),
                (width, length, height)
            ]
        else:
            # Full 6 permutations (can lay on any side)
            return [
                (length, width, height),
                (length, height, width),
                (width, length, height),
                (width, height, length),
                (height, length, width),
                (height, width, length)
            ]

    # Apply function
    parcels_df_sorted["Orientations"] = parcels_df_sorted.apply(
        lambda row: get_orientations(
            row["L (cm)"], row["W (cm)"], row["H (cm)"], row["Turnable"]
        ),
        axis=1
    )

    print("Parcels with orientations based on Turnable:")
    print(parcels_df_sorted[["ParcelID", "L (cm)", "W (cm)", "H (cm)", "Volume_cm3", "Turnable", "Orientations"]].head())

    def can_place_3D(parcel_dim, position, container_dim, placed_parcels, stackable, settings,
                 debug=False, parcel_id=None, container_id=None):
        """
        Checks if a parcel can be placed at a position inside a container.
        Returns (True/False, reason)
        debug: if True will print detailed reasons for rejection/acceptance
        parcel_id, container_id: optional identifiers for clearer debug prints
        """
        Lp, Wp, Hp = parcel_dim
        x, y, z = position
        Lc, Wc, Hc = container_dim

        def _hdr():
            pid = f"{parcel_id}" if parcel_id is not None else "?"
            cid = f"{container_id}" if container_id is not None else "?"
            return f"[Parcel {pid} | Container {cid} | Pos=({x},{y},{z}) | Orient=({Lp},{Wp},{Hp})]"

        # --- Check container boundaries ---
        if x + Lp > Lc or y + Wp > Wc or z + Hp > Hc:
            if debug:
                print(_hdr(), "➡️ Rejected: Out of bounds (container LxWxH = "
                    f"{Lc}x{Wc}x{Hc}, required extents = "
                    f"({x+Lp},{y+Wp},{z+Hp}))")
            return False, "Out of bounds"

        # --- Check overlap with already placed parcels ---
        for pp in placed_parcels:
            # defensive access of expected keys (older runs had KeyError 'dim')
            L2 = pp.get("L", pp.get("l", None))
            W2 = pp.get("W", pp.get("w", None))
            H2 = pp.get("H", pp.get("h", None))
            x2 = pp.get("X", pp.get("x", None))
            y2 = pp.get("Y", pp.get("y", None))
            z2 = pp.get("Z", pp.get("z", None))
            pid2 = pp.get("ParcelID", pp.get("Parcel_Id", "existing"))

            # if any necessary field missing, report and treat as overlap-safe (or decide)
            if None in (L2, W2, H2, x2, y2, z2):
                if debug:
                    print(_hdr(), f"⚠️ Skipping overlap check for existing entry (missing keys): {pp}")
                continue

            overlap_x = not (x + Lp <= x2 or x >= x2 + L2)
            overlap_y = not (y + Wp <= y2 or y >= y2 + W2)
            overlap_z = not (z + Hp <= z2 or z >= z2 + H2)

            if overlap_x and overlap_y and overlap_z:
                if debug:
                    print(_hdr(), f"➡️ Rejected: Overlap with placed parcel {pid2} at ({x2},{y2},{z2}) size ({L2},{W2},{H2})")
                return False, f"Overlap with {pid2}"

        # --- Check stackability rule ---
        if not stackable:
            # Check if parcel is being placed above something (z > 0)
            if z > 0:
                if debug:
                    print(_hdr(), "➡️ Rejected: Non-stackable parcel cannot be on top (z>0).")
                return False, "Non-stackable parcel cannot be on top"

        if debug:
            print(_hdr(), "✅ OK — fits here.")
        return True, "OK"

    def compute_score(utilization_pct, chargeable_wt, strategy, share_allowed):
        if strategy == "maximize_volume":
            return -utilization_pct  # higher utilization = better
        elif strategy == "minimize_cost":
            return chargeable_wt  # lower cost = better
        else:  # balanced
            if share_allowed:
                return chargeable_wt  # cost dominates
            else:
                # penalize low utilization and high cost
                return chargeable_wt - (utilization_pct * 0.5)

    def filter_placeable_parcels(parcels_df, containers_df):
        max_dims = containers_df[["L (cm)", "W (cm)", "H (cm)"]].max()
        max_payload = containers_df["Maximum Payload (kg)"].max()

        def fits(parcel):
            return (
                parcel["Weight (kg)"] <= max_payload and
                parcel["L (cm)"] <= max_dims["L (cm)"] and
                parcel["W (cm)"] <= max_dims["W (cm)"] and
                parcel["H (cm)"] <= max_dims["H (cm)"]
            )

        return parcels_df[parcels_df.apply(fits, axis=1)]

    import itertools
    import numpy as np
    
    containers_df = containers_df[containers_df["Qty"] > 0]
    def select_containers_for_shipment(
        parcels_df,
        containers_df,
        share_allowed=False,
        vol_divisor=6000,
        max_comb=3,
        topN_for_combos=12,
        strategy="balanced"
    ):
        # --- Step 1: Compute total cargo and volumetric weights ---
        total_cargo_vol = float(parcels_df["Volume_cm3"].sum())
        total_cargo_wt = float(parcels_df["Weight (kg)"].sum())
        volumetric_wt = total_cargo_vol / vol_divisor

        # --- Step 2: Build container info list ---
        container_info = []
        for _, c in containers_df.iterrows():
            tare_kg = float(c["Tare Weight (kg)"])
            vol_cm3 = float(c["Volume_cm3"])
            max_payload_kg = float(c["Maximum Payload (kg)"])

            gross_weight_kg = total_cargo_wt + tare_kg
            chargeable_weight_kg = max(volumetric_wt, gross_weight_kg)

            container_info.append({
                "ContainerID": c["ContainerID"],
                "vol_cm3": vol_cm3,
                "max_payload_kg": max_payload_kg,
                "tare_kg": tare_kg,
                "gross_weight_kg": gross_weight_kg,
                "volumetric_weight_kg": volumetric_wt,
                "chargeable_weight_kg": chargeable_weight_kg
            })

        # --- Stage 1: Feasible single containers ---
        feasible_single = []
        for c in container_info:
            if total_cargo_vol <= c["vol_cm3"] and total_cargo_wt <= c["max_payload_kg"]:
                feasible_single.append(c)

        # --- Stage 2: Try combinations ---
        container_info_sorted = sorted(container_info, key=lambda x: x["vol_cm3"], reverse=True)[:topN_for_combos]
        best_comb = None
        best_metrics = None
        best_util = 0
        best_chargeable_wt = float("inf")

        for r in range(2, max_comb + 1):
            for comb in itertools.combinations(container_info_sorted, r):
                total_vol = sum(c["vol_cm3"] for c in comb)
                total_max_payload = sum(c["max_payload_kg"] for c in comb)
                total_tare = sum(c["tare_kg"] for c in comb)

                gross_weight = total_cargo_wt + total_tare
                chargeable_weight = max(volumetric_wt, gross_weight)

                if total_cargo_vol <= total_vol and total_cargo_wt <= total_max_payload:
                    util = total_cargo_vol / total_vol * 100
                    if chargeable_weight < best_chargeable_wt:
                        best_chargeable_wt = chargeable_weight
                        best_util = util
                        best_comb = comb
                        best_metrics = {
                            "gross_weight_kg": gross_weight,
                            "volumetric_weight_kg": volumetric_wt,
                            "chargeable_weight_kg": chargeable_weight
                        }

        # --- Combine all valid options ---
        candidates = []

        for c in feasible_single:
            util = total_cargo_vol / c["vol_cm3"] * 100
            score = compute_score(util, c["chargeable_weight_kg"], strategy, share_allowed)
            candidates.append({
                "container_ids": [c["ContainerID"]],
                "total_vol_cm3": c["vol_cm3"],
                "utilization_pct": round(util, 2),
                "chargeable_weight_kg": c["chargeable_weight_kg"],
                "gross_weight_kg": c["gross_weight_kg"],
                "volumetric_weight_kg": c["volumetric_weight_kg"],
                "score": score,
                "reason": "Single container"
            })

        if best_comb:
            score = compute_score(best_util, best_metrics["chargeable_weight_kg"], strategy, share_allowed)
            candidates.append({
                "container_ids": [c["ContainerID"] for c in best_comb],
                "total_vol_cm3": sum(c["vol_cm3"] for c in best_comb),
                "utilization_pct": round(best_util, 2),
                **best_metrics,
                "score": score,
                "reason": f"Combination of {len(best_comb)} containers"
            })

        # --- Fallback: largest container ---
        largest = max(container_info, key=lambda x: x["vol_cm3"])
        if total_cargo_vol <= largest["vol_cm3"] and total_cargo_wt <= largest["max_payload_kg"]:
            util = total_cargo_vol / largest["vol_cm3"] * 100
            gross_weight = total_cargo_wt + largest["tare_kg"]
            chargeable_weight = max(volumetric_wt, gross_weight)
            score = compute_score(util, chargeable_weight, strategy, share_allowed)

            candidates.append({
                "container_ids": [largest["ContainerID"]],
                "total_vol_cm3": largest["vol_cm3"],
                "utilization_pct": round(util, 2),
                "chargeable_weight_kg": chargeable_weight,
                "gross_weight_kg": gross_weight,
                "volumetric_weight_kg": volumetric_wt,
                "score": score,
                "reason": "Fallback — largest container"
            })

        # --- Pick best candidate ---
        valid_candidates = [c for c in candidates if c["utilization_pct"] <= 100]
        if not valid_candidates:
            print("⚠️ No valid container or combination can fit all parcels. Proceeding with fallback container.")
            best = max(container_info, key=lambda x: x["vol_cm3"])
            return {
                "container_ids": [best["ContainerID"]],
                "total_container_vol_m3": best["vol_cm3"] / 1e6,
                "gross_weight_kg": best["gross_weight_kg"],
                "volumetric_weight_kg": best["volumetric_weight_kg"],
                "chargeable_weight_kg": best["chargeable_weight_kg"],
                "utilization_pct": round(total_cargo_vol / best["vol_cm3"] * 100, 2),
                "reason": "Fallback — largest container used despite overflow"
            }

        best = min(valid_candidates, key=lambda x: x["score"])

        print("Cargo volume (cm³):", total_cargo_vol)
        print("Selected container volume (cm³):", best["total_vol_cm3"])
        print("Utilization (%):", total_cargo_vol / best["total_vol_cm3"] * 100)

        return {
            "container_ids": best["container_ids"],
            "total_container_vol_m3": best["total_vol_cm3"] / 1e6,
            "gross_weight_kg": best["gross_weight_kg"],
            "volumetric_weight_kg": best["volumetric_weight_kg"],
            "chargeable_weight_kg": best["chargeable_weight_kg"],
            "utilization_pct": best["utilization_pct"],
            "reason": best["reason"]
        }

    # --- Strategy comparison: volume vs cost vs balanced ---
    for strategy in ["maximize_volume", "minimize_cost", "balanced"]:
        result = select_containers_for_shipment(
            parcels_df=parcels_df_sorted,
            containers_df=containers_df,
            share_allowed=False,
            vol_divisor=6000,
            max_comb=3,
            topN_for_combos=12,
            strategy=strategy
        )
        print(f"{strategy}: {result['utilization_pct']}% used, {result['chargeable_weight_kg']} kg charged")

    # -------------------------------
    # Step 0: Compute parcel chargeable weight
    # -------------------------------
    parcels_df_sorted["ChargeableWeight_kg"] = parcels_df_sorted[["Weight (kg)", "VolumetricWeight_kg"]].max(axis=1)

    # -------------------------------
    # Step 0.5: Compute parcel base area
    # -------------------------------
    parcels_df_sorted["BaseArea_cm2"] = parcels_df_sorted["L (cm)"] * parcels_df_sorted["W (cm)"]

    # -------------------------------
    # Primary sort: largest volume first; secondary: largest base area
    # -------------------------------

    parcels_df_sorted = parcels_df_sorted.sort_values(
        by=["Volume_cm3", "ChargeableWeight_kg", "BaseArea_cm2"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # --- CELL 2: call the selector ---
    filtered_df = filter_placeable_parcels(parcels_df, containers_df)
    share_allowed = False

    selector_result = select_containers_for_shipment(
        parcels_df=filter_placeable_parcels(parcels_df_sorted, containers_df),
        containers_df=containers_df,
        share_allowed=False,
        vol_divisor=6000,
        max_comb=3,
        topN_for_combos=12,
        strategy="balanced"
    )

    selected_container_ids = selector_result["container_ids"]
    containers_df_selected = containers_df[containers_df["ContainerID"].isin(selected_container_ids)].copy()

    # --------- Optimized 3D Best Fit loop with chargeable weight ----------
    assignments = []

    # Initialize container states
    container_3D_state = {cid: [] for cid in containers_df_selected["ContainerID"]}
    container_used_vol = {cid: 0 for cid in containers_df_selected["ContainerID"]}
    container_used_weight = {cid: 0.0 for cid in containers_df_selected["ContainerID"]}

    # Ensure Stackable column exists
    if "Stackable" not in parcels_df_sorted.columns:
        parcels_df_sorted["Stackable"] = True

    for idx, parcel in parcels_df_sorted.iterrows():
        best_fit = None
        min_waste = float("inf")
        remaining_parcels = len(parcels_df_sorted) - idx
        factor_remaining = 1 + (remaining_parcels / len(parcels_df_sorted))
        parcel_vol = float(parcel["Volume_cm3"])
        parcel_wt = float(parcel["ChargeableWeight_kg"])
        orientations = parcel["Orientations"]

        # --- Sort selected containers by available volume descending ---
        containers_df_sorted = containers_df_selected.copy()
        containers_df_sorted["available_vol"] = containers_df_sorted["Volume_cm3"] - containers_df_sorted["ContainerID"].map(container_used_vol)
        containers_df_sorted = containers_df_sorted.sort_values(by="available_vol", ascending=False)

        for _, container in containers_df_sorted.iterrows():
            cid = container["ContainerID"]
            container_dim = (container["L (cm)"], container["W (cm)"], container["H (cm)"])
            container_total_vol = float(container["Volume_cm3"])
            container_max_wt = float(container["Maximum Payload (kg)"])
            used_vol = container_used_vol.get(cid, 0)
            used_wt = container_used_weight.get(cid, 0)
            available_vol = container_total_vol - used_vol
            available_wt = container_max_wt - used_wt
            placed_parcels = container_3D_state[cid]

            # Skip if parcel cannot fit by volume or weight
            if parcel_vol > available_vol or parcel_wt > available_wt:
                continue

            for orientation in orientations:
                Lp, Wp, Hp = orientation

                # --- Generate candidate positions ---
                # Wall clearance
                wall_L = float(settings.get("wall_clearance_L_cm", 0))
                wall_W = float(settings.get("wall_clearance_W_cm", 0))
                wall_H = float(settings.get("wall_clearance_H_cm", 0))
                
                # Inter-box clearance
                inter_L = float(settings.get("inter_box_clearance_L_cm", 0))
                inter_W = float(settings.get("inter_row_clearance_W_cm", 0))
                candidate_positions = [(wall_L, wall_W, 0)]
                for existing in placed_parcels:
                    ex, ey, ez = existing["X"], existing["Y"], existing["Z"]
                    eL, eW, eH = existing["L"], existing["W"], existing["H"]

                    candidate_positions.append((ex + eL + inter_L, ey, ez))
                    candidate_positions.append((ex, ey + eW + inter_W, ez))
                    # --- stacking logic ---
                    if existing["Stackable"]:
                        candidate_positions.append((ex, ey, ez + eH))
                    else:
                        # Non-stackable: only place above if weight <= 32 kg
                        if parcel_wt <= 32:
                            candidate_positions.append((ex, ey, ez + eH))

                # --- Test candidate positions ---
                for pos in candidate_positions:
                    can_place, reason = can_place_3D(
                        orientation, pos, container_dim, placed_parcels, parcel["Stackable"], settings,
                        debug=False, parcel_id=parcel["ParcelID"], container_id=cid
                    )

                    if can_place:
                        leftover_after = container_total_vol - (used_vol + parcel_vol)
                        weighted_leftover = leftover_after * factor_remaining

                        if weighted_leftover < min_waste:
                            min_waste = weighted_leftover
                            best_fit = {
                                "ContainerID": cid,
                                "Orientation": orientation,
                                "Position": tuple(map(int, pos)),
                                "LeftoverAfter": leftover_after,
                                "ContainerTotalVol": container_total_vol
                            }
                        break

        # --- Place parcel if a fit was found ---
        container_brand_map = dict(zip(containers_df["ContainerID"], containers_df["Brand"]))
        if best_fit:
            cid = best_fit["ContainerID"]
            container_3D_state[cid].append({
                "ParcelID": parcel["ParcelID"],
                "X": best_fit["Position"][0],
                "Y": best_fit["Position"][1],
                "Z": best_fit["Position"][2],
                "L": best_fit["Orientation"][0],
                "W": best_fit["Orientation"][1],
                "H": best_fit["Orientation"][2],
                "Stackable": parcel["Stackable"],
                "ChargeableWeight_kg": parcel_wt,
                "Weight_kg": parcel["Weight (kg)"]
            })
            container_used_vol[cid] += parcel_vol
            container_used_weight[cid] += parcel_wt

            current_leftover = best_fit["ContainerTotalVol"] - container_used_vol[cid]
            assignments.append({
                "ParcelID": parcel["ParcelID"],
                "Length_cm": parcel["L (cm)"],
                "Width_cm": parcel["W (cm)"],
                "Height_cm": parcel["H (cm)"],
                "Weight_kg": parcel["Weight (kg)"],
                "Brand": parcel["Brand"],
                "ContainerID": cid,
                "Container Brand":container_brand_map.get(cid, "Unknown"),
                "Orientation": best_fit["Orientation"],
                "Position": best_fit["Position"],
                "LeftoverVolume": current_leftover,
                "ContainerChargeableUsed": container_used_weight[cid]
            })
        else:
            print(f"⚠️ No fit found for {parcel['ParcelID']}")
            assignments.append({
                "ParcelID": parcel["ParcelID"],
                "Length_cm": parcel["L (cm)"],
                "Width_cm": parcel["W (cm)"],
                "Height_cm": parcel["H (cm)"],
                "Weight_kg": parcel["Weight (kg)"],
                "Brand": parcel["Brand"],
                "ContainerID": None,
                "ContainerID": None,
                "Orientation": None,
                "Position": None,
                "LeftoverVolume": None,
                "ContainerChargeableUsed": None
            })

    # --- Convert leftover volume from cm^3 to m^3 ---
    assignments_df = pd.DataFrame(assignments)
    if "LeftoverVolume" in assignments_df.columns:
        assignments_df["LeftoverVolume_m3"] = (assignments_df["LeftoverVolume"] / 1e6).round(3)
        assignments_df = assignments_df.drop(columns=["LeftoverVolume"])

    # Round container used weight for display
    if "ContainerChargeableUsed" in assignments_df.columns:
        assignments_df["ContainerChargeableUsed"] = assignments_df["ContainerChargeableUsed"].round(2)

    print("Optimized Best Fit assignments (3D placement with chargeable weight, dynamic scenarios, base-area):")
    print(assignments_df.head(60))

    container_summary = []
    for cid in selected_container_ids:  # use only selected containers
        # Volume calculations
        used_vol_cm3 = container_used_vol.get(cid, 0)
        total_vol_cm3 = containers_df.loc[containers_df["ContainerID"] == cid, "Volume_cm3"].values[0]
        used_vol_m3 = used_vol_cm3 / 1e6
        total_vol_m3 = total_vol_cm3 / 1e6
        utilization_vol_pct = (used_vol_m3 / total_vol_m3 * 100) if total_vol_m3 else 0

        # Weight calculations — using actual weight
        used_wt = sum(parcel["Weight_kg"] for parcel in container_3D_state[cid])
        max_wt = containers_df.loc[containers_df["ContainerID"] == cid, "Maximum Payload (kg)"].values[0]
        utilization_wt_pct = (used_wt / max_wt * 100) if max_wt else 0

        container_summary.append({
            "ContainerID": cid,
            "UsedVolume_m3": round(used_vol_m3, 3),
            "TotalVolume_m3": round(total_vol_m3, 3),
            "VolumeUtilization_%": round(utilization_vol_pct, 2),
            "UsedWeight_kg": round(used_wt, 2),
            "MaxPayload_kg": round(max_wt, 2),
            "WeightUtilization_%": round(utilization_wt_pct, 2),
            "ParcelsCount": len(container_3D_state[cid])
        })

    container_summary_df = pd.DataFrame(container_summary)
    print(container_summary_df)   

    unplaceable_df = parcels_df_sorted[~parcels_df_sorted["ParcelID"].isin(
        filter_placeable_parcels(parcels_df_sorted, containers_df)["ParcelID"]
    )]

    print(f"🚫 Unplaceable parcels: {len(unplaceable_df)}")
    if not unplaceable_df.empty:
        display(unplaceable_df[["ParcelID", "L (cm)", "W (cm)", "H (cm)", "Weight (kg)"]])
      
    # --- Export both DataFrames to Excel ---
    output_path = "optimization_results.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        assignments_df.to_excel(writer, index=False, sheet_name="ParcelAssignments")
        container_summary_df.to_excel(writer, index=False, sheet_name="ContainerSummary")

    print(f"✅ Exported results to {output_path}")

    return assignments_df
