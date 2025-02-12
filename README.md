## This is the repo of DebateCoder (currently under review at ACL'25)

### Prepare Data
We have got two dataset, APPS and CodeContest. APPS has three difficulty levels, i.e. introductory, interview and competition. CodeContest is splited into two difficulty levels, according to their difficulty index.

Before processing data, raw dataset files need to be downloaded.

To process apps data, run:
```
python preprocess_apps.py [level] [YOUR APPS ROOT DATA PATH]
```
where level can be intro, inter and comp. This will yields a .gz file in ```../data/``` folder.

To process CodeContest data, run:
```
python preprocess_codecontest.py --language python --data_path ../data --difficulty hard --data_root_path /home/jzchen/ML/Code/data/CodeContest,
```

### Run DebateCoder
```
CUDA_VISIBLE_DEVICES=5 python main.py -A claude-3-5-sonnet-20241022 -B gpt-4o-mini  -l python -d appsnew -e 10 -a 0 -b 1 --dataset CodeContest --cur_exp_name claude-4o-simple-test-codecontest-easy --start_idx 0 --end_idx 200  --level easy

```
where ```-A```  and ```-B``` denote debating model A and B, ```cur_exp_name``` is the folder name that the results will be stored at.

### Test 
First, fill your API key at ```/util/config.py``` before running DebateCoder.

To test code CodeContest result, suppose your results is at ```claude-4o-simple-test-codecontest-hard``` run:
```
python calculate_codecontest.py claude-4o-simple-test-codecontest-hard 0 63 final claude-3-5-sonnet-20241022 gpt-4o-mini
```

To test APPS result, suppose your results is at ```claude-4o-medium-test``` run:
```
python calculate_apps.py claude-4o-medium-test 1 100 final claude-3-5-sonnet-20241022 gpt-4o-mini
```
