link_id_in_cac_list_mask = last_event_df.link_id.isin(cac_df.link_id)
merge_df = last_event_df[link_id_in_cac_list_mask].merge(cac_df, on='link_id', how='left')

#Keep only valid associations (first_occurence between lp_inizio and lp_fine)
mask = merge_df.first_occurrence >= merge_df.lp_inizio 
mask &= merge_df.first_occurrence < merge_df.lp_fine
merge_df = merge_df[mask]

#Associated alarm_id to more than one lp_id and lp_type
alarm_id_to_lavoro_ids_df = merge_df.groupby(['alarm_id']).lp_id.unique().to_frame().reset_index(drop=False)
alarm_id_to_tipi_lavoro_df = merge_df.groupby(['alarm_id']).lp_type.unique().to_frame().reset_index(drop=False) 
alarm_id_to_lavoro_ids_and_tipi_lavoro_df = alarm_id_to_lavoro_ids_df.merge(alarm_id_to_tipi_lavoro_df, on='alarm_id', how='left')

#Final association alarm_id to lp_id 
last_event_df = last_event_df.merge(alarm_id_to_lavoro_ids_and_tipi_lavoro_df, on='alarm_id', how='left')
last_event_df = last_event_df.reset_index(drop=True)
last_event_df.head()