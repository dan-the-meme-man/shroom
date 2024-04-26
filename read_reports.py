import os
import re

pattern = re.compile(r'\s+')

hparams_dirs = [os.path.join('reports', x) for x in os.listdir('reports')]

best_acc = 0
best_macro_prec = 0
best_macro_rec = 0
best_macro_f1 = 0
best_micro_prec = 0
best_micro_rec = 0
best_micro_f1 = 0

for hparams_dir in hparams_dirs:
    hparams_files = [os.path.join(hparams_dir, x) for x in os.listdir(hparams_dir)]
    for hparams_file in hparams_files:
        epoch = os.path.basename(hparams_file).split('_')[-1].split('.')[0]
        #print(f'epoch: {epoch}')
        with open(hparams_file, 'r') as f:
            lines = f.readlines()[16:19]
            info = re.split(pattern, ''.join(lines))
            #print(info)
            acc = float(info[2])
            macro_prec = float(info[6])
            macro_rec = float(info[7])
            macro_f1 = float(info[8])
            micro_prec = float(info[12])
            micro_rec = float(info[13])
            micro_f1 = float(info[14])
            
            if acc > best_acc:
                best_acc = acc
                best_acc_epoch = epoch
                best_acc_hparams = hparams_file
            
            if macro_prec > best_macro_prec:
                best_macro_prec = macro_prec
                best_macro_prec_epoch = epoch
                best_macro_prec_hparams = hparams_file
                
            if macro_rec > best_macro_rec:
                best_macro_rec = macro_rec
                best_macro_rec_epoch = epoch
                best_macro_rec_hparams = hparams_file
            
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_macro_f1_epoch = epoch
                best_macro_f1_hparams = hparams_file
            
            if micro_prec > best_micro_prec:
                best_micro_prec = micro_prec
                best_micro_prec_epoch = epoch
                best_micro_prec_hparams = hparams_file
                
            if micro_rec > best_micro_rec:
                best_micro_rec = micro_rec
                best_micro_rec_epoch = epoch
                best_micro_rec_hparams = hparams_file
                
            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                best_micro_f1_epoch = epoch
                best_micro_f1_hparams = hparams_file
                
def fmt(metric, value, epoch, hparams):
    print(f'Best {metric}: {value} at epoch {epoch} with hyperparameters {hparams}\n')

fmt('accuracy', best_acc, best_acc_epoch, best_acc_hparams)
fmt('macro precision', best_macro_prec, best_macro_prec_epoch, best_macro_prec_hparams)
fmt('macro recall', best_macro_rec, best_macro_rec_epoch, best_macro_rec_hparams)
fmt('macro f1', best_macro_f1, best_macro_f1_epoch, best_macro_f1_hparams)
fmt('micro precision', best_micro_prec, best_micro_prec_epoch, best_micro_prec_hparams)
fmt('micro recall', best_micro_rec, best_micro_rec_epoch, best_micro_rec_hparams)
fmt('micro f1', best_micro_f1, best_micro_f1_epoch, best_micro_f1_hparams)