"""Download Vestibular-Schwannoma-SEG (TCIA, via NCI IDC): contrast T1 MR + RTSTRUCT contours."""
from pathlib import Path
from idc_index import index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "raw" / "vs_seg"
OUT.mkdir(parents=True, exist_ok=True)

c = index.IDCClient()
# One contrast-enhanced T1 series per patient (largest if several), plus every RTSTRUCT.
t1 = c.sql_query("""
SELECT PatientID, SeriesInstanceUID, SeriesDescription FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY PatientID ORDER BY instanceCount DESC) rk
  FROM index WHERE collection_id='vestibular_schwannoma_seg' AND Modality='MR'
    AND lower(SeriesDescription) LIKE 't1%') WHERE rk=1""")
rt = c.sql_query("""SELECT PatientID, SeriesInstanceUID FROM index
  WHERE collection_id='vestibular_schwannoma_seg' AND Modality='RTSTRUCT'""")
t1.to_csv(OUT / "t1_series.csv", index=False)
print(f"T1 series: {len(t1)} patients; RTSTRUCT series: {len(rt)}")
uids = t1.SeriesInstanceUID.tolist() + rt.SeriesInstanceUID.tolist()
c.download_from_selection(seriesInstanceUID=uids, downloadDir=str(OUT),
                          dirTemplate="%PatientID/%Modality_%SeriesInstanceUID", show_progress_bar=False)
print("done")
