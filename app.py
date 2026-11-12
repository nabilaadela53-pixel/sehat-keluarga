"""
SehatKeluarga - prototipe aplikasi konsultasi gejala awal berbasis keluarga (Streamlit).

Jalankan (pilih salah satu):
    pip install -r requirements.txt
    python app.py              <- cukup dengan Python, peramban terbuka otomatis
    streamlit run app.py       <- cara standar Streamlit

Catatan penting:
- Hasil adalah INFORMASI AWAL berdasarkan data, bukan diagnosis.
- Model Decision Tree dilatih langsung dari dataset Peduli Sehat (Peduli_Sehat.xls) sebagai
  prototipe. Pada tahap praktik berikutnya model dapat diganti dengan hasil RapidMiner.
- Pemetaan penyakit -> tingkat kewaspadaan, daftar tanda bahaya, dan isi artikel edukasi di bawah
  hanyalah CONTOH dan harus divalidasi tenaga kesehatan sebelum dipakai.
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# --- Peluncur: memungkinkan aplikasi dijalankan dengan perintah "python app.py" ---
if __name__ == "__main__":
    from streamlit.runtime import exists as _streamlit_running
    if not _streamlit_running():
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:])
        sys.exit(0)

st.set_page_config(page_title="SehatKeluarga", page_icon="🩺", layout="centered", initial_sidebar_state="expanded")

# =====================================================================
# KONFIGURASI
# =====================================================================
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Peduli_Sehat.xls")
CONF_THRESHOLD = 0.60      # di bawah ini sistem mengajukan pertanyaan lanjutan
MAX_QUESTIONS = 4          # jumlah maksimum pertanyaan lanjutan
MIN_QUESTIONS = 1          # minimal 1 pertanyaan lanjutan selalu ditampilkan (bila masih ada gejala untuk ditanyakan)
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# Kolom yang bukan gejala (faktor risiko / duplikat / kosong) dikeluarkan dari model
RISK_COLS = ["hubungan_di_luar_nikah", "riwayat_konsumsi_alkohol", "pernah_transfusi_darah",
             "menerima_suntikan_tidak_steril", "riwayat_keluarga"]

# Perbaikan nama label pada dataset
LABEL_FIX = {
    "Hepatitis 100": "Hepatitis C", "Hepatitis B.": "Hepatitis B", "Hepatitis D.": "Hepatitis D",
    "Hepatitis e": "Hepatitis E", "Peptik Ulcer Diseae": "Tukak lambung",
    "Hemmorhoids dimorfik (tumpukan)": "Wasir (hemoroid)", "Pembuluh mekar": "Varises",
    "(Vertigo) Paroymsal Positional Vertigo": "Vertigo (BPPV)", "Osteoarthristis": "Osteoarthritis",
    "Kelumpuhan (perdarahan otak)": "Stroke (perdarahan otak)",
}

# CONTOH pemetaan tingkat kewaspadaan (selain dua daftar ini = Sedang). Validasi oleh nakes!
LEVEL_RENDAH = {"Flu biasa", "Alergi", "Infeksi jamur", "Jerawat"}
LEVEL_TINGGI = {"Demam berdarah", "Penyakit tipus", "Malaria", "TBC", "Radang paru-paru",
                "Serangan jantung", "Stroke (perdarahan otak)", "AIDS", "Hepatitis E",
                "Hepatitis alkoholik", "Asma bronkial"}
# CONTOH daftar tanda bahaya (kolom dataset). Bila ada -> tingkat kewaspadaan Tinggi
RED_FLAGS = ["nyeri_dada", "sesak_napas", "kesadaran_berubah", "koma", "kelemahan_satu_sisi_tubuh",
             "bicara_pelo", "darah_dalam_dahak", "pendarahan_lambung"]

# Gejala umum pada UI -> kolom dataset
COMMON = {"Demam": "demam_tinggi", "Batuk": "batuk", "Pilek": "hidung_meler",
          "Sakit Kepala": "sakit_kepala", "Mual": "mual", "Muntah": "muntah",
          "Nyeri Otot": "nyeri_otot", "Diare": "diare", "Lemas": "lesu"}

SARAN = {
    "Rendah": ["Istirahat dan minum cukup cairan", "Pantau suhu tubuh dan gejala secara berkala",
               "Periksakan diri bila keluhan memburuk atau berlanjut"],
    "Sedang": ["Istirahat dan minum cukup cairan", "Pantau suhu tubuh secara berkala",
               "Segera ke IGD bila muncul tanda bahaya"],
    "Tinggi": ["Segera ke IGD atau fasilitas kesehatan terdekat",
               "Jangan menunda pemeriksaan", "Minta anggota keluarga mendampingi"],
}
BANNER = {  # (latar, teks, kalimat anjuran)
    "Rendah": ("#DCFCE7", "#14532D", "Umumnya dapat dirawat mandiri sambil dipantau."),
    "Sedang": ("#F2E86D", "#3B3300", "Disarankan periksa ke puskesmas / klinik."),
    "Tinggi": ("#FEE2E2", "#7F1D1D", "Segera ke IGD / fasilitas kesehatan terdekat."),
}
BADGE = {"Rendah": "#22C55E", "Sedang": "#F5A623", "Tinggi": "#EF4444"}

ARTIKEL = [  # CONTOH konten: ganti dengan materi yang sudah divalidasi tenaga kesehatan
    ("Penyakit", "Mengenali gejala demam berdarah",
     "Demam tinggi mendadak, nyeri di belakang mata, nyeri otot dan sendi, mual, serta ruam dapat menjadi tanda. "
     "Bila disertai nyeri perut hebat, muntah terus-menerus, atau perdarahan, segera ke fasilitas kesehatan."),
    ("Ke Dokter", "Kapan anak demam perlu ke dokter?",
     "Segera periksakan bila anak tampak sangat lemas, sulit dibangunkan, kejang, sesak napas, muntah terus-menerus, "
     "atau demam tidak membaik dalam beberapa hari."),
    ("Penyakit", "Merawat anggota keluarga yang flu",
     "Perbanyak istirahat dan cairan, jaga kebersihan tangan, dan pisahkan alat makan. "
     "Periksakan bila muncul sesak napas atau keluhan memburuk."),
    ("Pencegahan", "Menjaga kebersihan tangan dan lingkungan",
     "Cuci tangan dengan sabun, tutup tempat penampungan air, dan jaga kebersihan rumah untuk mengurangi penularan penyakit."),
    ("Ke Dokter", "Tanda bahaya yang perlu segera ke IGD",
     "Nyeri dada, sesak napas, kesadaran menurun, kelemahan satu sisi tubuh, bicara pelo, atau muntah/dahak berdarah."),
]

# =====================================================================
# DATA DAN MODEL
# =====================================================================
def pretty(col: str) -> str:
    return col.replace("_", " ").capitalize()


def clean_label(x: str) -> str:
    return LABEL_FIX.get(x, x)


def demo_dataframe() -> pd.DataFrame:
    """Data mini cadangan bila file dataset tidak ditemukan (hanya untuk uji tampilan)."""
    prof = {
        "Flu biasa": ["hidung_tersumbat", "hidung_meler", "bersin_terus_menerus", "batuk", "nyeri_otot", "sakit_kepala", "meriang"],
        "Alergi": ["bersin_terus_menerus", "mata_berair", "meriang", "menggigil"],
        "Demam berdarah": ["demam_tinggi", "nyeri_di_belakang_mata", "sakit_kepala", "mual", "nyeri_otot", "ruam_kulit", "muntah"],
        "Penyakit tipus": ["demam_tinggi", "kelelahan", "diare", "sembelit", "mual", "sakit_kepala", "nyeri_perut"],
        "GERD": ["asam_lambung", "sakit_perut", "muntah", "batuk", "sariawan_di_lidah"],
        "Gastroenteritis": ["diare", "muntah", "dehidrasi", "mata_cekung"],
        "Migrain": ["sakit_kepala", "mual", "penglihatan_kabur_dan_terganggu", "leher_kaku", "mudah_marah"],
        "Serangan jantung": ["nyeri_dada", "sesak_napas", "muntah", "berkeringat"],
    }
    cols = sorted({s for v in prof.values() for s in v})
    rng = np.random.default_rng(0)
    rows = []
    for d, sym in prof.items():
        for _ in range(60):
            r = {c: 0 for c in cols}
            for s in sym:
                r[s] = int(rng.random() < 0.9)
            r["prognosis"] = d
            rows.append(r)
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    demo = False
    try:
        df = pd.read_excel(DATA_FILE)
    except Exception:
        df, demo = demo_dataframe(), True
    drop = [c for c in df.columns if c.startswith("tidak_bernama") or c.endswith(".1") or c in RISK_COLS]
    df = df.drop(columns=drop)
    y = df.pop("prognosis").astype(str).map(clean_label)
    feats = list(df.columns)
    # Augmentasi: pengguna biasanya hanya melaporkan sebagian gejala, sehingga data latih ditambah
    # salinan dengan gejala yang dihilangkan secara acak (peluang gejala dipertahankan = 0.5).
    X0 = df[feats].values
    rng = np.random.default_rng(42)
    Xs, ys = [X0], [y.values]
    for _ in range(4):
        Xs.append(X0 * (rng.random(X0.shape) < 0.5))
        ys.append(y.values)
    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=25, random_state=42)
    clf.fit(pd.DataFrame(np.vstack(Xs), columns=feats), np.concatenate(ys))
    return clf, feats, demo


CLF, FEATS, DEMO_MODE = load_model()
ALL_LABELS = {pretty(c): c for c in FEATS}


def level_of(disease: str, red: list) -> str:
    if red or disease in LEVEL_TINGGI:
        return "Tinggi"
    return "Rendah" if disease in LEVEL_RENDAH else "Sedang"


def run_analysis():
    S = st.session_state
    vec = pd.Series(0.0, index=FEATS)
    for c in S.sel:
        if c in vec.index:
            vec[c] = 1
    for c, v in S.answers.items():
        vec[c] = v
    X = pd.DataFrame([vec.values], columns=FEATS)
    proba = CLF.predict_proba(X)[0]
    order = np.argsort(proba)[::-1][:3]
    top = [(str(CLF.classes_[i]), float(proba[i])) for i in order if proba[i] > 0]
    # pertanyaan lanjutan: gejala pertama pada jalur pohon yang belum diketahui pengguna
    known = set(S.sel) | set(S.answers)
    nxt = None
    for n in CLF.decision_path(X).indices:
        f = CLF.tree_.feature[n]
        if f >= 0 and FEATS[f] not in known:
            nxt = FEATS[f]
            break
    red = [c for c in RED_FLAGS if c in vec.index and vec[c] == 1]
    return {"top": top, "conf": top[0][1], "next": nxt, "red": red}


def route(res):
    S = st.session_state
    S.res = res
    perlu_tanya = res["conf"] < CONF_THRESHOLD or S.qcount < MIN_QUESTIONS
    if perlu_tanya and S.qcount < MAX_QUESTIONS and res["next"]:
        S.cur_q, S.page = res["next"], "lanjut"
    else:
        S.page = "hasil"


# =====================================================================
# STATE
# =====================================================================
S = st.session_state
S.setdefault("page", "beranda")
S.setdefault("members", [{"nama": "Ibu", "usia": 38}, {"nama": "Ayah", "usia": 40},
                         {"nama": "Raka", "usia": 7}, {"nama": "Nenek", "usia": 68}])
S.setdefault("member_idx", 2)
S.setdefault("sel", [])
S.setdefault("umum", [])
S.setdefault("cari", [])
S.setdefault("durasi", "3")
S.setdefault("answers", {})
S.setdefault("qcount", 0)
S.setdefault("cur_q", None)
S.setdefault("res", None)
S.setdefault("saved", True)
S.setdefault("history", [
    {"tgl": "20 Okt 2026", "nama": "Raka", "gejala": "Demam, sakit kepala, mual", "hasil": "Demam berdarah", "level": "Sedang"},
    {"tgl": "2 Sep 2026", "nama": "Raka", "gejala": "Batuk, pilek, demam", "hasil": "Flu", "level": "Rendah"},
    {"tgl": "14 Jul 2026", "nama": "Raka", "gejala": "Demam, nyeri otot, lemas", "hasil": "Tipes", "level": "Sedang"},
])


def goto(p):
    S.page = p


def member_label(m):
    return f"{m['nama']} ({m['usia']} Tahun)"


# =====================================================================
# GAYA (CSS)
# =====================================================================
st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility:hidden;}
.block-container {max-width:440px; padding:0 1rem 1.2rem 1rem;}
.sk-head {background:#0F766E; color:#fff; text-align:center; font-weight:700; font-size:1.2rem;
          padding:2.4rem 1rem 1.3rem; margin:0 -1rem 1rem -1rem;}
.sk-hello {font-weight:700; font-size:1.15rem; margin:0;}
.sk-sub {color:#6B7280; font-size:.8rem; margin:0 0 .6rem 0;}
.sk-avatars {display:flex; gap:14px; margin:.4rem 0 1rem 0;}
.sk-av {width:38px; height:38px; border-radius:50%; background:#E5E7EB; color:#0F766E; font-weight:700;
        display:flex; align-items:center; justify-content:center; font-size:.85rem;}
.sk-av.on {background:#0F766E; color:#fff;}
.sk-chip {display:inline-block; background:#E5E7EB; color:#111827; border-radius:999px; padding:3px 12px;
          font-size:.72rem; font-weight:600; margin:2px 4px 2px 0;}
.sk-card {background:#fff; border:1px solid #E5E7EB; border-radius:14px; padding:12px 14px; margin-bottom:10px;}
.sk-q {background:#fff; border-radius:12px; padding:18px 14px 6px; text-align:center; font-weight:700; margin:8px 0;}
.sk-info {background:#0F766E; color:#fff; border-radius:16px; padding:14px; text-align:center; margin-top:14px;}
.sk-info small {display:block; font-size:.72rem; opacity:.9; margin-top:4px;}
.sk-warn {background:#FEF3C7; border:1px solid #F59E0B; color:#78350F; border-radius:12px; padding:10px;
          text-align:center; font-size:.72rem; margin-top:14px;}
.sk-banner {border-radius:12px; padding:12px 14px; margin-bottom:10px; font-weight:600; font-size:.9rem;}
.sk-label {color:#6B7280; font-size:.85rem; font-weight:600; margin:10px 0 4px;}
.sk-barrow {display:flex; justify-content:space-between; font-weight:700; margin-top:8px;}
.sk-track {height:8px; background:#9CA3AF; border-radius:6px; overflow:hidden;}
.sk-fill {height:8px; background:#0F766E;}
.sk-badge {float:right; color:#fff; font-size:.65rem; font-weight:700; padding:2px 10px; border-radius:999px;}
.sk-date {color:#9CA3AF; font-size:.78rem;}
.sk-photo {background:#EFEFEF; border-radius:8px; width:84px; height:84px; display:flex; align-items:center;
           justify-content:center; color:#6B7280; font-size:.8rem; float:left; margin-right:12px;}
.sk-ok {color:#0F766E; font-weight:600; font-size:.75rem;}
div.stButton > button, div.stLinkButton > a {border-radius:999px; font-weight:600; width:100%;}
div.stButton > button[kind="primary"], div.stLinkButton > a[kind="primary"] {background:#0F766E; border-color:#0F766E; color:#fff;}
.sk-navgap {margin-top:1.2rem; border-top:2px solid #CBD5E1; padding-top:.6rem;}
</style>
""", unsafe_allow_html=True)


def header(title):
    st.markdown(f'<div class="sk-head">{title}</div>', unsafe_allow_html=True)


def bottom_nav():
    active = {"beranda": "beranda", "cek": "cek", "lanjut": "cek", "hasil": "cek",
              "riwayat": "riwayat", "edukasi": "edukasi"}.get(S.page, "")
    st.markdown('<div class="sk-navgap"></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("beranda", "Beranda"), ("cek", "Cek Gejala"),
                                         ("riwayat", "Riwayat"), ("edukasi", "Edukasi")]):
        col.button(label, key=f"nav_{key}", on_click=goto, args=(key,),
                   type="primary" if active == key else "secondary", use_container_width=True)


# =====================================================================
# HALAMAN
# =====================================================================
def page_beranda():
    header("SehatKeluarga")
    st.markdown('<p class="sk-hello">Halo, Bu Sari</p><p class="sk-sub">Siapa yang perlu diperiksa hari ini?</p>',
                unsafe_allow_html=True)
    av = "".join(f'<div class="sk-av{" on" if i == S.member_idx else ""}">{m["nama"][0].upper()}</div>'
                 for i, m in enumerate(S.members))
    st.markdown(f'<div class="sk-avatars">{av}<div class="sk-av">+</div></div>', unsafe_allow_html=True)
    st.button("Cek Gejala Sekarang", type="primary", on_click=goto, args=("cek",), use_container_width=True)
    st.write("")
    c1, c2 = st.columns(2)
    c1.button("Riwayat Gejala", on_click=goto, args=("riwayat",), use_container_width=True)
    c2.button("Edukasi Kesehatan", on_click=goto, args=("edukasi",), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.button("Faskes Terdekat", on_click=goto, args=("faskes",), use_container_width=True)
    c4.button("Profil Keluarga", on_click=goto, args=("profil",), use_container_width=True)
    st.markdown('<div class="sk-warn">Informasi bersifat awal dan bukan<br>pengganti diagnosis tenaga kesehatan.</div>',
                unsafe_allow_html=True)
    bottom_nav()


def page_cek():
    header("Cek Gejala")
    labels = [member_label(m) for m in S.members]
    S.setdefault("w_member", labels[min(S.member_idx, len(labels) - 1)])
    if S.w_member not in labels:
        S.w_member = labels[0]
    st.caption("Untuk siapa?")
    pilih = st.selectbox("Untuk siapa?", labels, key="w_member", label_visibility="collapsed")
    S.member_idx = labels.index(pilih)

    st.caption("Cari gejala")
    S.setdefault("w_cari", S.cari)
    cari = st.multiselect("Cari gejala", list(ALL_LABELS), key="w_cari", placeholder="Cari gejala....",
                          label_visibility="collapsed")
    S.cari = cari

    st.markdown('<div class="sk-label">Gejala Umum</div>', unsafe_allow_html=True)
    S.setdefault("w_umum", S.umum)
    umum = st.pills("Gejala umum", list(COMMON), selection_mode="multi", key="w_umum", label_visibility="collapsed")
    S.umum = umum or []

    sel = list(dict.fromkeys([COMMON[u] for u in S.umum] + [ALL_LABELS[c] for c in S.cari]))
    st.markdown(f'<div class="sk-label">{len(sel)} gejala dipilih</div>', unsafe_allow_html=True)

    st.markdown('<div class="sk-label">Sudah berapa hari?</div>', unsafe_allow_html=True)
    S.setdefault("w_durasi", S.durasi)
    durasi = st.segmented_control("Durasi", ["1", "2", "3", "4+"], key="w_durasi", label_visibility="collapsed")
    S.durasi = durasi or S.durasi

    st.write("")
    if st.button("Analisis Gejala", type="primary", use_container_width=True):
        if not sel:
            st.warning("Pilih minimal 1 gejala terlebih dahulu.")
        else:
            S.sel, S.answers, S.qcount, S.saved = sel, {}, 0, False
            route(run_analysis())
            st.rerun()
    bottom_nav()


def page_lanjut():
    header("Pertanyaan Lanjutan")
    n = min(S.qcount + 1, MAX_QUESTIONS)
    st.markdown(f'<div class="sk-sub">Pertanyaan {n} dari sekitar {MAX_QUESTIONS}</div>', unsafe_allow_html=True)
    st.progress(n / MAX_QUESTIONS)
    chips = "".join(f'<span class="sk-chip">{pretty(c)}</span>' for c in S.sel)
    st.markdown(f'<div class="sk-label">Gejala yang sudah dipilih</div>{chips}', unsafe_allow_html=True)
    nama = S.members[S.member_idx]["nama"]
    st.markdown(f'<div class="sk-q">Apakah {nama} juga<br>mengalami {pretty(S.cur_q).lower()}?</div>',
                unsafe_allow_html=True)

    def answer(v):
        S.answers[S.cur_q] = v
        S.qcount += 1
        route(run_analysis())

    c1, c2, c3 = st.columns(3)
    c1.button("Ya", type="primary", on_click=answer, args=(1,), use_container_width=True)
    c2.button("Tidak", on_click=answer, args=(0,), use_container_width=True)
    c3.button("Tidak tahu", on_click=answer, args=(0,), use_container_width=True)
    d1, d2 = st.columns(2)
    d1.button("< Kembali", on_click=goto, args=("cek",), use_container_width=True)
    d2.button("Lewati dan lihat hasil >", on_click=goto, args=("hasil",), use_container_width=True)
    st.markdown('<div class="sk-info"><b>Mengapa Ditanya?</b><small>Pertanyaan yang dipilih sistem<br>'
                'agar hasil informasi lebih tepat.</small></div>', unsafe_allow_html=True)
    bottom_nav()


def page_hasil():
    header("Hasil Informasi Awal")
    res = S.res or run_analysis()
    top = res["top"]
    level = level_of(top[0][0], res["red"])
    bg, fg, kalimat = BANNER[level]
    st.markdown(f'<div class="sk-banner" style="background:{bg};color:{fg}">Tingkat Kewaspadaan : {level.upper()}'
                f'<br><span style="font-weight:500">{kalimat}</span></div>', unsafe_allow_html=True)
    if res["red"]:
        st.error("Terdapat gejala tanda bahaya: " + ", ".join(pretty(c).lower() for c in res["red"]) +
                 ". Segera cari pertolongan medis.")
    st.markdown('<div class="sk-label">Kemungkinan penyakit ( contoh ilustrasi )</div>', unsafe_allow_html=True)
    for nama, p in top:
        pct = round(p * 100)
        st.markdown(f'<div class="sk-barrow"><span>{nama}</span><span>{pct}%</span></div>'
                    f'<div class="sk-track"><div class="sk-fill" style="width:{pct}%"></div></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="sk-label">Saran tindak lanjut</div>', unsafe_allow_html=True)
    saran = list(SARAN[level])
    if S.durasi == "4+" and level != "Tinggi":
        saran.append("Keluhan sudah 4 hari atau lebih, sebaiknya periksakan diri")
    for s in saran:
        st.markdown(f"- {s}")

    if not S.saved:  # simpan ke riwayat satu kali
        d = date.today()
        S.history.insert(0, {"tgl": f"{d.day} {BULAN[d.month - 1]} {d.year}",
                             "nama": S.members[S.member_idx]["nama"],
                             "gejala": ", ".join(pretty(c) for c in S.sel),
                             "hasil": top[0][0], "level": level})
        S.saved = True

    st.link_button("Cari Faskes Terdekat", "https://www.google.com/maps/search/puskesmas+terdekat",
                   type="primary", use_container_width=True)
    st.button("Lihat Riwayat", on_click=goto, args=("riwayat",), use_container_width=True)
    st.caption("Bukan pengganti diagnosis tenaga kesehatan")
    bottom_nav()


def page_riwayat():
    header("Riwayat Gejala")
    nama = [m["nama"] for m in S.members]
    S.setdefault("w_filter", "Raka" if "Raka" in nama else "Semua")
    if S.w_filter not in ["Semua"] + nama:
        S.w_filter = "Semua"
    filt = st.pills("Filter", ["Semua"] + nama, key="w_filter", label_visibility="collapsed") or "Semua"
    data = [h for h in S.history if filt == "Semua" or h["nama"] == filt]
    if not data:
        st.info("Belum ada riwayat.")
    for h in data:
        st.markdown(
            f'<div class="sk-card"><span class="sk-badge" style="background:{BADGE[h["level"]]}">{h["level"].upper()}</span>'
            f'<span class="sk-date">{h["tgl"]}{"" if filt != "Semua" else " - " + h["nama"]}</span><br><br>'
            f'<b style="font-size:.85rem">{h["gejala"]}</b><br><br>'
            f'<span class="sk-ok">Hasil: {h["hasil"]}</span></div>', unsafe_allow_html=True)
    bottom_nav()


def page_edukasi():
    header("Edukasi Kesehatan")
    q = st.text_input("Cari artikel", placeholder="Cari artikel...", label_visibility="collapsed")
    kat = st.pills("Kategori", ["Penyakit", "Pencegahan", "Ke Dokter"], key="w_kat", label_visibility="collapsed")
    for k, judul, isi in ARTIKEL:
        if (kat and k != kat) or (q and q.lower() not in judul.lower()):
            continue
        st.markdown(f'<div class="sk-card" style="overflow:hidden"><div class="sk-photo">Foto</div>'
                    f'<b>{judul}</b><br><br><span class="sk-ok">Divalidasi Tenaga Kesehatan</span></div>',
                    unsafe_allow_html=True)
        with st.expander("Baca"):
            st.write(isi)
            st.caption("Contoh konten: ganti dengan materi yang sudah divalidasi tenaga kesehatan.")
    bottom_nav()


def page_faskes():
    header("Faskes Terdekat")
    st.write("Cari puskesmas, klinik, atau rumah sakit terdekat melalui peta.")
    st.link_button("Puskesmas terdekat", "https://www.google.com/maps/search/puskesmas+terdekat",
                   type="primary", use_container_width=True)
    st.link_button("Klinik terdekat", "https://www.google.com/maps/search/klinik+terdekat", use_container_width=True)
    st.link_button("IGD / Rumah sakit terdekat", "https://www.google.com/maps/search/igd+rumah+sakit+terdekat",
                   use_container_width=True)
    st.button("< Kembali", on_click=goto, args=("beranda",))
    bottom_nav()


def page_profil():
    header("Profil Keluarga")
    for m in S.members:
        st.markdown(f'<div class="sk-card"><b>{m["nama"]}</b> <span class="sk-date">({m["usia"]} Tahun)</span></div>',
                    unsafe_allow_html=True)
    with st.expander("+ Tambah anggota"):
        nama = st.text_input("Nama")
        usia = st.number_input("Usia (tahun)", 0, 120, 30)
        if st.button("Simpan anggota", type="primary") and nama.strip():
            S.members.append({"nama": nama.strip(), "usia": int(usia)})
            S.pop("w_member", None)
            st.rerun()
    st.button("< Kembali", on_click=goto, args=("beranda",))
    bottom_nav()


# ---------- Navigasi 6 tampilan (sidebar) ----------
NAV6 = {"1. Beranda": "beranda", "2. Cek Gejala": "cek", "3. Pertanyaan Lanjutan": "lanjut",
        "4. Hasil Informasi Awal": "hasil", "5. Riwayat Gejala": "riwayat", "6. Edukasi Kesehatan": "edukasi"}
LABEL_BY_PAGE = {v: k for k, v in NAV6.items()}


def _nav_change():
    S.page = NAV6[S.w_nav]


def ensure_demo_data():
    """Agar halaman 3 dan 4 bisa dibuka langsung dari sidebar, gunakan contoh gejala bila belum ada analisis."""
    if S.page in ("lanjut", "hasil") and S.res is None:
        S.sel, S.answers, S.qcount, S.saved = ["demam_tinggi", "sakit_kepala", "mual"], {}, 0, True
        S.res = run_analysis()
    if S.page == "lanjut" and not S.cur_q:
        S.cur_q = (S.res or run_analysis())["next"] or "nyeri_otot"


S.w_nav = LABEL_BY_PAGE.get(S.page, "1. Beranda")
with st.sidebar:
    st.markdown("### 6 Tampilan Aplikasi")
    st.radio("Pilih tampilan", list(NAV6), key="w_nav", on_change=_nav_change)
    st.caption("Halaman 3 dan 4 juga muncul otomatis setelah menekan Analisis Gejala. "
               "Jika dibuka langsung dari sini, dipakai contoh gejala: demam, sakit kepala, mual.")
ensure_demo_data()

PAGES = {"beranda": page_beranda, "cek": page_cek, "lanjut": page_lanjut, "hasil": page_hasil,
         "riwayat": page_riwayat, "edukasi": page_edukasi, "faskes": page_faskes, "profil": page_profil}
PAGES.get(S.page, page_beranda)()

if DEMO_MODE:
    st.caption("Mode demo: file Peduli_Sehat.xls tidak ditemukan, memakai data mini.")
