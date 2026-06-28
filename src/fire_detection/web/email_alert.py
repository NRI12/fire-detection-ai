import base64
import json
import smtplib
import ssl
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "email_config.json"


class GmailSender:
    def load_config(self) -> dict:
        defaults = {"sender_email": "", "app_password": "", "recipient_email": ""}
        if CONFIG_PATH.exists():
            try:
                defaults.update(json.loads(CONFIG_PATH.read_text()))
            except Exception:
                pass
        return defaults

    def save_config(self, updates: dict) -> None:
        current = self.load_config()
        current.update(updates)
        CONFIG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2))

    def is_configured(self) -> bool:
        cfg = self.load_config()
        return bool(cfg.get("sender_email") and cfg.get("app_password") and cfg.get("recipient_email"))

    def test_connection(self) -> tuple[bool, str]:
        cfg = self.load_config()
        if not cfg.get("sender_email") or not cfg.get("app_password"):
            return False, "Chưa nhập email và App Password"
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                smtp.login(cfg["sender_email"], cfg["app_password"])
            return True, "Kết nối thành công"
        except smtplib.SMTPAuthenticationError:
            return False, "Sai email hoặc App Password"
        except Exception as e:
            return False, str(e)

    def send_alert(self, detections: list, output_b64: str, gradcam_b64: str, source: str, inference_ms: float) -> bool:
        cfg = self.load_config()
        if not cfg.get("sender_email") or not cfg.get("app_password") or not cfg.get("recipient_email"):
            return False

        fires = [d for d in detections if d["class_name"] == "fire"]
        smokes = [d for d in detections if d["class_name"] == "smoke"]
        timestamp = time.strftime("%d/%m/%Y %H:%M:%S")

        rows = "".join(
            f"<tr>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{i + 1}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd;color:{'#e63946' if d['class_name'] == 'fire' else '#6e7681'};font-weight:bold'>{d['class_name']}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{d['confidence'] * 100:.1f}%</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd;font-size:12px'>{', '.join(map(str, d['box']))}</td>"
            f"</tr>"
            for i, d in enumerate(detections)
        )

        html = f"""<!DOCTYPE html>
<html><body style="font-family:sans-serif;background:#f5f5f5;color:#222;padding:24px;max-width:620px;margin:0 auto">
<div style="background:#e63946;color:#fff;padding:16px 20px;border-radius:8px 8px 0 0">
  <h2 style="margin:0;font-size:18px">CẢNH BÁO: PHÁT HIỆN LỬA / KHÓI</h2>
</div>
<div style="background:#fff;padding:20px;border:1px solid #e0e0e0;border-top:none;border-radius:0 0 8px 8px">
  <table style="font-size:13px;margin-bottom:16px;width:100%">
    <tr><td style="color:#666;padding:3px 12px 3px 0;width:130px">Thời gian:</td><td><b>{timestamp}</b></td></tr>
    <tr><td style="color:#666;padding:3px 12px 3px 0">Nguồn:</td><td>{source}</td></tr>
    <tr><td style="color:#666;padding:3px 12px 3px 0">Thời gian xử lý:</td><td>{inference_ms} ms</td></tr>
    <tr><td style="color:#666;padding:3px 12px 3px 0">Lửa (fire):</td><td><b style="color:#e63946">{len(fires)}</b></td></tr>
    <tr><td style="color:#666;padding:3px 12px 3px 0">Khói (smoke):</td><td><b style="color:#6e7681">{len(smokes)}</b></td></tr>
  </table>
  <h3 style="color:#444;font-size:13px;margin:16px 0 6px;border-bottom:1px solid #eee;padding-bottom:4px">Chi tiết phát hiện:</h3>
  <table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px">
    <thead style="background:#f5f5f5">
      <tr>
        <th style="padding:6px 8px;border:1px solid #ddd;text-align:left">#</th>
        <th style="padding:6px 8px;border:1px solid #ddd;text-align:left">Loại</th>
        <th style="padding:6px 8px;border:1px solid #ddd;text-align:left">Độ tin cậy</th>
        <th style="padding:6px 8px;border:1px solid #ddd;text-align:left">Vùng phát hiện</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#666;font-size:12px;margin-bottom:6px"><b>Ảnh kết quả phát hiện (Bounding Box):</b></p>
  <img src="cid:output_img" style="max-width:100%;border-radius:6px;border:1px solid #ddd;display:block;margin-bottom:16px">
  <p style="color:#666;font-size:12px;margin-bottom:6px"><b>Bản đồ nhiệt GradCAM (EigenCAM):</b></p>
  <img src="cid:gradcam_img" style="max-width:100%;border-radius:6px;border:1px solid #ddd;display:block">
</div>
</body></html>"""

        msg = MIMEMultipart("related")
        msg["Subject"] = f"[CẢNH BÁO] {len(fires)} lửa, {len(smokes)} khói - {timestamp}"
        msg["From"] = cfg["sender_email"]
        msg["To"] = cfg["recipient_email"]

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)

        for cid, b64_data in [("output_img", output_b64), ("gradcam_img", gradcam_b64)]:
            img = MIMEImage(base64.b64decode(b64_data), "jpeg")
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline")
            msg.attach(img)

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(cfg["sender_email"], cfg["app_password"])
            smtp.sendmail(cfg["sender_email"], cfg["recipient_email"], msg.as_bytes())
        return True
