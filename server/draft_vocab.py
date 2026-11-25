"""Launch-time draft vocabulary policy; independent of GPU architecture."""


def configure(mode, family, env):
    if mode is None:
        return  # preserve explicit low-level environment overrides
    if mode not in ("full", "restricted"):
        raise ValueError("draft vocabulary must be full or restricted")
    if family not in ("qwen38-27b", "qwen38-flash-next"):
        raise ValueError("--draft-vocab requires Qwen3.8-27B or Flash-Next")
    full = mode == "full"
    if family == "qwen38-27b":
        env.update(QE_DRAFT_VOCAB="0" if full else "98304",
                   QE_MTP_VOCAB="0" if full else "65536",
                   QE_DF_HEAD="full" if full else "")
    else:
        env["FLASH_DRAFT_V"] = "0" if full else "65536"
