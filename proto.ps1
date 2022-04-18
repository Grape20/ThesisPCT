# A to D
Start-Transcript
cd Proto_DA-ours 
python proto.py ../data/office31 -s A -t D --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-ours-cos 
python proto.py ../data/office31 -s A -t D --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-master
python proto.py ../data/office31 -s A -t D --epochs 32 -i 10 -p 5 -beta 0.001 --auto_bs True
cd .. 
Stop-Transcript

# A to W
Start-Transcript
cd Proto_DA-ours 
python proto.py ../data/office31 -s A -t W --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-ours-cos 
python proto.py ../data/office31 -s A -t W --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-master
python proto.py ../data/office31 -s A -t W --epochs 32 -i 10 -p 5 -beta 0.001 --auto_bs True
cd .. 
Stop-Transcript

# D to A
Start-Transcript
cd Proto_DA-ours 
python proto.py ../data/office31 -s D -t A --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-ours-cos 
python proto.py ../data/office31 -s D -t A --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-master
python proto.py ../data/office31 -s D -t A --epochs 32 -i 10 -p 5 -beta 0.001 --auto_bs True
cd .. 
Stop-Transcript

# D to W
Start-Transcript
cd Proto_DA-ours 
python proto.py ../data/office31 -s D -t W --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-ours-cos 
python proto.py ../data/office31 -s D -t W --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-master
python proto.py ../data/office31 -s D -t W --epochs 32 -i 10 -p 5 -beta 0.001 --auto_bs True
cd .. 
Stop-Transcript

# W to A
Start-Transcript
cd Proto_DA-ours 
python proto.py ../data/office31 -s W -t A --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-ours-cos 
python proto.py ../data/office31 -s W -t A --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-master
python proto.py ../data/office31 -s W -t A --epochs 32 -i 10 -p 5 -beta 0.001 --auto_bs True
cd .. 
Stop-Transcript

# W to D
Start-Transcript
cd Proto_DA-ours 
python proto.py ../data/office31 -s W -t D --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-ours-cos 
python proto.py ../data/office31 -s W -t D --epochs 32 -i 10 -p 5 --auto_bs True
cd ..\Proto_DA-master
python proto.py ../data/office31 -s W -t D --epochs 32 -i 10 -p 5 -beta 0.001 --auto_bs True
cd .. 
Stop-Transcript




