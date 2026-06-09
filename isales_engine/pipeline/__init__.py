"""Dual-LLM AI pipeline: one main LLM streams the reply + N parallel referee
LLMs (each emits a bare category token) gate it before audio is released; an
offline extractor runs post-call. (The old 3-layer role-PK → judges → polish
design is gone.)"""
