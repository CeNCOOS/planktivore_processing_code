```mermaid
flowchart LR

I[(Thalassa/planktivore/YYYY-MM-DD-LRAH-di/)]
J1[setup_process_planktivore_highmag_apr2024.json]
P1[file_format_refactor.py]
OI[(Thalassa/DeepSea-AI/data/planktivore/raw/)]
J2[setup_process_planktivore_highmag_apr2024parquet.json]
P2[names_to_parquet.py]
O1[April_2024_Ahi_highmag.parquet]
P3[read_planktivore_camlogs.py]


   subgraph "Reformat Images"
      direction TB
        J1-->P1
   end
   subgraph "Write Image names to file"
      direction TB
         J2-->P2
   end
   I-->"Reformat Images"-->OI
   I-->"Write Image names to file"-->O1
```