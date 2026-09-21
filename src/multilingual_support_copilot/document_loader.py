from pathlib import Path

def load_text_file(dosya_yolu: Path) ->str:
    

    try: 
        return dosya_yolu.read_text(encoding="utf-8")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Hata: {dosya_yolu} dosyası bulunamadı.")
    except PermissionError:
        raise PermissionError(
            f"Hata: '{dosya_yolu}' dosyasını okumak için gerekli yetkiniz yok.")
