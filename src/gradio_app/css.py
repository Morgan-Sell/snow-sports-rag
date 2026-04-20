"""Custom CSS for the alpine Gradio frontend.

The CSS layers on top of :func:`make_alpine_theme` and handles what themes
can't: card-based layout for the Sources panel, frost/glass header, and
compact Debug tables.
"""

ALPINE_CSS = """
:root {
  --alpine-snow: #F8FAFC;
  --alpine-frost: #EEF4F8;
  --alpine-ice: #D8E2EC;
  --alpine-glacier: #1E6FA9;
  --alpine-deep: #0E3A5F;
  --alpine-pine: #2F6A4A;
  --alpine-slate: #475569;
  --alpine-slate-2: #1E293B;
}

/* ------- Header ------- */
.alpine-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 18px 22px;
  border-radius: 14px;
  background: linear-gradient(135deg, #F3F8FB 0%, #E2ECF3 100%);
  border: 1px solid var(--alpine-ice);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  margin-bottom: 16px;
}
.alpine-header .alpine-mark {
  font-size: 26px;
  line-height: 1;
}
.alpine-header .alpine-title {
  font-weight: 600;
  font-size: 18px;
  color: var(--alpine-slate-2);
  letter-spacing: 0.2px;
}
.alpine-header .alpine-subtitle {
  color: var(--alpine-slate);
  font-size: 13px;
  margin-top: 2px;
}

/* ------- Source cards ------- */
.alpine-sources {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.alpine-sources .alpine-empty {
  color: var(--alpine-slate);
  font-size: 13px;
  padding: 14px;
  border: 1px dashed var(--alpine-ice);
  border-radius: 12px;
  background: var(--alpine-snow);
  text-align: center;
}
.alpine-card {
  border: 1px solid var(--alpine-ice);
  border-radius: 12px;
  padding: 12px 14px;
  background: #FFFFFF;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
  transition: box-shadow 120ms ease, transform 120ms ease;
}
.alpine-card:hover {
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
  transform: translateY(-1px);
}
.alpine-card .alpine-card-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.alpine-card .alpine-badge {
  display: inline-flex;
  min-width: 22px;
  justify-content: center;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--alpine-glacier);
  color: #FFFFFF;
  font-size: 12px;
  font-weight: 600;
}
.alpine-card .alpine-doc {
  font-weight: 600;
  color: var(--alpine-slate-2);
  font-size: 14px;
  overflow-wrap: anywhere;
}
.alpine-card .alpine-section {
  color: var(--alpine-slate);
  font-size: 12px;
  margin-top: 1px;
}
.alpine-card .alpine-score {
  margin-left: auto;
  color: var(--alpine-pine);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  white-space: nowrap;
}
.alpine-card .alpine-hint {
  margin-left: auto;
  color: var(--alpine-slate);
  font-size: 13px;
  line-height: 1;
  opacity: 0.55;
  transition: opacity 120ms ease, color 120ms ease;
  cursor: help;
}
.alpine-card:hover .alpine-hint {
  opacity: 1;
  color: var(--alpine-glacier);
}
.alpine-card .alpine-snippet {
  color: var(--alpine-slate-2);
  font-size: 13px;
  line-height: 1.45;
  margin-top: 4px;
}

/* ------- Settings read-only info ------- */
.alpine-info {
  display: grid;
  grid-template-columns: max-content 1fr;
  column-gap: 16px;
  row-gap: 6px;
  padding: 12px 14px;
  background: var(--alpine-frost);
  border: 1px solid var(--alpine-ice);
  border-radius: 12px;
  font-size: 13px;
}
.alpine-info dt {
  color: var(--alpine-slate);
  font-weight: 500;
}
.alpine-info dd {
  color: var(--alpine-slate-2);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin: 0;
  overflow-wrap: anywhere;
}

/* ------- Debug ------- */
.alpine-debug h4 {
  margin: 12px 0 6px;
  font-size: 13px;
  color: var(--alpine-slate);
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.alpine-debug table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.alpine-debug th, .alpine-debug td {
  border-bottom: 1px solid var(--alpine-ice);
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}
.alpine-debug th {
  color: var(--alpine-slate);
  font-weight: 600;
}
.alpine-debug .alpine-latency {
  display: inline-flex;
  gap: 10px;
  flex-wrap: wrap;
  padding: 6px 0 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--alpine-slate-2);
}
.alpine-debug .alpine-latency span {
  background: var(--alpine-frost);
  border: 1px solid var(--alpine-ice);
  border-radius: 999px;
  padding: 2px 10px;
}

/* ------- Evidence banner ------- */
.alpine-banner {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 12px;
  margin-bottom: 12px;
  border: 1px solid;
  font-size: 13px;
  line-height: 1.45;
}
.alpine-banner .alpine-banner-icon {
  font-size: 18px;
  line-height: 1.2;
  flex-shrink: 0;
}
.alpine-banner .alpine-banner-title {
  font-weight: 600;
  margin-bottom: 2px;
}
.alpine-banner-warn {
  background: #F2F7EE;
  border-color: #CFE1C2;
  color: var(--alpine-pine);
}
.alpine-banner-warn .alpine-banner-title { color: #1F4A33; }
.alpine-banner-warn .alpine-banner-body { color: var(--alpine-slate-2); }

/* ------- Empty-state hero ------- */
.alpine-hero {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  padding: 18px 20px;
  border-radius: 14px;
  border: 1px solid var(--alpine-ice);
  background: linear-gradient(180deg, #FFFFFF 0%, #F3F8FB 100%);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  margin-bottom: 10px;
}
.alpine-hero-mark {
  font-size: 20px;
  color: var(--alpine-glacier);
  line-height: 1;
}
.alpine-hero-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--alpine-slate-2);
}
.alpine-hero-sub {
  font-size: 13px;
  color: var(--alpine-slate);
  max-width: 56ch;
}
.alpine-hero-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 2px;
}

/* ------- Misc ------- */
footer { display: none !important; }
"""
