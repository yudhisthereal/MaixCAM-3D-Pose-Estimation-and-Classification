def get_fall_info(online_targets_det, online_targets, index, fallParam, queue_size, fps):
    fall_down = False

    # Get current and previous bounding boxes
    cur_bbox = [online_targets_det.x, online_targets_det.y, online_targets_det.w, online_targets_det.h]

    if online_targets["bbox"][index].empty():
        return False

    pre_bbox = online_targets["bbox"][index].get()
    _ = online_targets["points"][index].get()  # keep points queue in sync with bbox

    elapsed_ms = queue_size * 1000 / fps if fps > 0 else queue_size * 1000

    # 1. Vertical speed of top (y) coordinate — downward movement = positive
    dy_top = cur_bbox[1] - pre_bbox[1]
    v_top = dy_top / elapsed_ms

    # 2. Vertical change of height (shrinking = falling)
    dh = pre_bbox[3] - cur_bbox[3]
    v_height = dh / elapsed_ms

    print(f"[DEBUG] v_top = {v_top:.6f}, v_height = {v_height:.6f}, threshold = {fallParam['v_bbox_y']}")

    if v_top > fallParam["v_bbox_y"] or v_height > fallParam["v_bbox_y"]:
        print(f"[FALL] Fall triggered (top drop or height shrink): v_top = {v_top:.6f}, v_height = {v_height:.6f}")
        fall_down = True

    return fall_down
