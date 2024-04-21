import zipfile
import pandas as pd
from association_alarm_id_to_lp_id import associate_alarm_id_to_lp_id


if __name__ == "__main__":
    
    # Percorso del file ZIP da estrarre
    zip_mob_file_path = "alarms datasets/mob/20230101-20240101_inpas_mob_preprocess__an__last_event__last_event__ext1.zip"
    zip_tx_adsl_file_path = "alarms datasets/tx/20230101-20240101_inpas_tx_preprocess__adsl__last_event__last_event__ext1.zip"
    zip_tx_pdh_file_path = "alarms datasets/tx/20230101-20240101_inpas_tx_preprocess__pdh__last_event__last_event__ext1.zip"
    zip_tx_ptn_file_path = "alarms datasets/tx/20230101-20240101_inpas_tx_preprocess__ptn__last_event__last_event__ext1.zip"

    # Percorso della cartella di destinazione per l'estrazione
    extract_to_folder_mob = "alarms datasets/mob"
    extract_to_folder_tx = "alarms datasets/tx"

    # Percorso del file ZIP da estrarre
    zip_files_paths = {
        zip_mob_file_path: extract_to_folder_mob,
        zip_tx_adsl_file_path: extract_to_folder_tx,
        zip_tx_pdh_file_path: extract_to_folder_tx,
        zip_tx_ptn_file_path: extract_to_folder_tx,
    }

    # Aprire il file ZIP
    for zip_file_path, extract_to_folder in zip_files_paths.items():
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Estrarre tutto il contenuto nella cartella di destinazione
            zip_ref.extractall(extract_to_folder)
            
    print("Files extracted successfully!")       
            
    mob_alarms_df = pd.read_parquet('alarms datasets/mob/20230101-20240101_inpas_mob_preprocess__an__last_event__last_event__ext1.parquet')

    # tx_alarms_adsl_df = pd.read_parquet('alarms datasets/tx/20230101-20240101_inpas_tx_preprocess__adsl__last_event__last_event__ext1.parquet')
    # tx_alarms_pdh_df = pd.read_parquet('alarms datasets/tx/20230101-20240101_inpas_tx_preprocess__pdh__last_event__last_event__ext1.parquet')
    # tx_alarms_ptn_df = pd.read_parquet('alarms datasets/tx/20230101-20240101_inpas_tx_preprocess__ptn__last_event__last_event__ext1.parquet')

    # print("Files loaded successfully!")
    
    # filtered_mob_alarms_df = mob_alarms_df.drop(columns=['lp_type', 'lp_id'])


    # # filtered_tx_alarms_adsl_df = tx_alarms_adsl_df.drop(columns=['lp_type', 'lp_id'])
    # # filtered_tx_alarms_pdh_df = tx_alarms_pdh_df.drop(columns=['lp_type', 'lp_id'])
    # # filtered_tx_alarms_ptn_df = tx_alarms_ptn_df.drop(columns=['lp_type', 'lp_id'])

    # print("Columns dropped successfully!")

    # lavori_programmati_df = pd.read_csv("lavori programmati/20230101_20240101_export_cac_all_preprocessed.csv")
    
    # print("Lavori programmati loaded successfully!")

    # new_mob_alarms_adsl_df = associate_alarm_id_to_lp_id(filtered_mob_alarms_df, lavori_programmati_df)
    
    
    # print(new_mob_alarms_adsl_df.head())