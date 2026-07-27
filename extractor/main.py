from collections import Counter

try:
    from .models import MirpExtractor, carica_pazienti, load_configuration
except ImportError:
    from models import MirpExtractor, carica_pazienti, load_configuration


def main() -> None:
    # Carica e valida i parametri scritti in config_extractor.yaml.
    config = load_configuration()

    # Cerca le coppie CT + RTStruct e associa lo stato microsatellitare dal CSV.
    pazienti = carica_pazienti(config)
    print(f"Pazienti caricati: {len(pazienti)}")

    if not pazienti:
        print("Nessun paziente valido trovato.")
        return

    if config.n_test is not False:
        numero_pazienti = len(pazienti)
        pazienti = pazienti[: config.n_test]
        print(
            "Modalità n-test attiva: "
            f"analizzo i primi {len(pazienti)} pazienti su {numero_pazienti}."
        )

    conteggio_stati = Counter(
        paziente.stato_microsatellitare.value
        for paziente in pazienti
        if paziente.stato_microsatellitare is not None
    )
    print(
        "Stato microsatellitare caricato: "
        + ", ".join(
            f"{stato}={quantita}"
            for stato, quantita in sorted(conteggio_stati.items())
        )
    )

    pazienti_senza_stato = [
        paziente.nome
        for paziente in pazienti
        if paziente.stato_microsatellitare is None
    ]
    if pazienti_senza_stato:
        print(
            "Attenzione: stato microsatellitare non disponibile per "
            f"{len(pazienti_senza_stato)} pazienti."
        )

    # Crea l'estrattore (Wrapper di MIRP) usando i parametri della configurazione.
    mirp_config = config.mirp
    if not mirp_config.export_features:
        print(
            "Nota: export_features era false; lo imposto a true per scrivere "
            "il CSV finale aggregato."
        )
        mirp_config = mirp_config.model_copy(update={"export_features": True})

    extractor = MirpExtractor(config=mirp_config)
    output_dir = config.data.ensure_output_dir()

    print(
        f"Avvio l'estrazione radiomica per {len(pazienti)} pazienti "
        "in modalità sequenziale."
    )
    risultati = extractor.extract_batch(
        pazienti,
        output_dir=output_dir,
        verbose=True,
    )

    numero_tabelle = sum(
        len(feature_tables or [])
        for feature_tables in risultati.values()
    )
    print(
        f"Estrazione completata: {len(risultati)} pazienti, "
        f"{numero_tabelle} tabelle restituite."
    )

    csv_path = config.data.ensure_features_csv_path()
    numero_righe = extractor.write_feature_csv(
        risultati=risultati,
        pazienti=pazienti,
        csv_path=csv_path,
    )
    print(f"Feature scritte in: {csv_path} ({numero_righe} righe).")


if __name__ == "__main__":
    main()
