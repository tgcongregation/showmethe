# Getting Transaction Data CSV

1. Go to reports section in Wave
2. Run the Account Transactions for the desired date range
3. Export to CSV
4. Since it isn't a _real_ CSV, run:
   ```/home/edl/git/the_grove/wave_utils/extract_wave_transaction_report.sh `pwd`/ FILE```
5. This creates FILE_enhanced.csv
