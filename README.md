# Getting Transaction Data CSV

1. Go to reports section in Wave
2. Run the Account Transactions for the desired date range
3. Export to CSV, saving it to wave_export.csv
4. Since it isn't a _real_ CSV, run:
   ```/home/edl/git/the_grove/wave_utils/extract_wave_transaction_report.sh `pwd`/ FILE```
5. This creates FILE_enhanced.csv

# Run Jenkins

Run Jenkins locally to do the above as well as deploy. Save the file to /home/edl/work/personal/church-accounting/wave_export.csv