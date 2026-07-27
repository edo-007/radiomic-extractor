try:
    from .models import carica_pazienti, load_configuration
except ImportError:
    from models import carica_pazienti, load_configuration


# Carica e valida i parametri scritti in config_extractor.yaml.
config = load_configuration()

# Cerca nelle cartelle dei dati le coppie CT + RTStruct.
pazienti = carica_pazienti(config)
print(f"Pazienti caricati: {len(pazienti)}")


def ordina_pazienti_per_data(
    lista_pazienti: list,
    use_rt: bool = False,
    piu_recenti_prima: bool = False,
) -> list:
    """Ordina una lista di pazienti in base alla StudyDate dei loro file DICOM.

    Di default ordina dal più vecchio al più recente (cronologico ascendente).
    """
    # Usiamo sorted() con una funzione lambda come chiave di ordinamento.
    # p.get_study_date(use_rt) restituirà la stringa 'YYYYMMDD' usata per il confronto.
    return sorted(
        lista_pazienti,
        key=lambda p: p.get_study_date(use_rt=use_rt),
        reverse=piu_recenti_prima,
    )

for paz in ordina_pazienti_per_data(pazienti, piu_recenti_prima=True):
    # print(f" {paz.nome} {paz.get_study_date()}")
    print(f" {paz.nome} {paz.get_patient_id()}")

if not pazienti:
    print("Nessun paziente valido trovato.")
    exit()


