from collections import Counter

from models import MirpExtractor, load_configuration, carica_pazienti


def main() -> None:
    # Carica e valida i parametri scritti in config.yaml.
    config = load_configuration()

    # Cerca le coppie CT + RTStruct e associa lo stato microsatellitare dal CSV.
    pazienti = carica_pazienti(config)
    print(f"Pazienti caricati: {len(pazienti)}")

    if not pazienti:
        print("Nessun paziente valido trovato.")
        return

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
    extractor = MirpExtractor(config=mirp_config)

    print(
        f"Avvio l'estrazione radiomica per {len(pazienti)} pazienti "
        "in modalità sequenziale."
    )
    risultati = extractor.extract_batch(
        pazienti,
        output_dir=config.data.ensure_output_dir(),
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


if __name__ == "__main__":
    main()
