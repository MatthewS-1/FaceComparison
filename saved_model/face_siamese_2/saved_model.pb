їщ
═Б
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.12v2.3.0-54-gfcc4b966f18╣Ѕ
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	ђ*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
Z
rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namerho
S
rho/Read/ReadVariableOpReadVariableOprho*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
б
sequential/encoder_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!sequential/encoder_conv1/kernel
Џ
3sequential/encoder_conv1/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv1/kernel*&
_output_shapes
:*
dtype0
њ
sequential/encoder_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namesequential/encoder_conv1/bias
І
1sequential/encoder_conv1/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv1/bias*
_output_shapes
:*
dtype0
б
sequential/encoder_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!sequential/encoder_conv2/kernel
Џ
3sequential/encoder_conv2/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv2/kernel*&
_output_shapes
: *
dtype0
њ
sequential/encoder_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namesequential/encoder_conv2/bias
І
1sequential/encoder_conv2/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv2/bias*
_output_shapes
: *
dtype0
б
sequential/encoder_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*0
shared_name!sequential/encoder_conv3/kernel
Џ
3sequential/encoder_conv3/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv3/kernel*&
_output_shapes
: 0*
dtype0
њ
sequential/encoder_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_namesequential/encoder_conv3/bias
І
1sequential/encoder_conv3/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv3/bias*
_output_shapes
:0*
dtype0
б
sequential/encoder_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*0
shared_name!sequential/encoder_conv4/kernel
Џ
3sequential/encoder_conv4/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv4/kernel*&
_output_shapes
:0@*
dtype0
њ
sequential/encoder_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namesequential/encoder_conv4/bias
І
1sequential/encoder_conv4/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv4/bias*
_output_shapes
:@*
dtype0
б
sequential/encoder_conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@P*0
shared_name!sequential/encoder_conv5/kernel
Џ
3sequential/encoder_conv5/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv5/kernel*&
_output_shapes
:@P*
dtype0
њ
sequential/encoder_conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_namesequential/encoder_conv5/bias
І
1sequential/encoder_conv5/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv5/bias*
_output_shapes
:P*
dtype0
ї
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђ
ђ*(
shared_namesequential/dense/kernel
Ё
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel* 
_output_shapes
:
ђ
ђ*
dtype0
Ѓ
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_namesequential/dense/bias
|
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes	
:ђ*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:╚*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:╚*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:╚*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:╚*
dtype0
Љ
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*+
shared_nameRMSprop/dense_1/kernel/rms
і
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes
:	ђ*
dtype0
ѕ
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_1/bias/rms
Ђ
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:*
dtype0
║
+RMSprop/sequential/encoder_conv1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+RMSprop/sequential/encoder_conv1/kernel/rms
│
?RMSprop/sequential/encoder_conv1/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv1/kernel/rms*&
_output_shapes
:*
dtype0
ф
)RMSprop/sequential/encoder_conv1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)RMSprop/sequential/encoder_conv1/bias/rms
Б
=RMSprop/sequential/encoder_conv1/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv1/bias/rms*
_output_shapes
:*
dtype0
║
+RMSprop/sequential/encoder_conv2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+RMSprop/sequential/encoder_conv2/kernel/rms
│
?RMSprop/sequential/encoder_conv2/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv2/kernel/rms*&
_output_shapes
: *
dtype0
ф
)RMSprop/sequential/encoder_conv2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)RMSprop/sequential/encoder_conv2/bias/rms
Б
=RMSprop/sequential/encoder_conv2/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv2/bias/rms*
_output_shapes
: *
dtype0
║
+RMSprop/sequential/encoder_conv3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*<
shared_name-+RMSprop/sequential/encoder_conv3/kernel/rms
│
?RMSprop/sequential/encoder_conv3/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv3/kernel/rms*&
_output_shapes
: 0*
dtype0
ф
)RMSprop/sequential/encoder_conv3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*:
shared_name+)RMSprop/sequential/encoder_conv3/bias/rms
Б
=RMSprop/sequential/encoder_conv3/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv3/bias/rms*
_output_shapes
:0*
dtype0
║
+RMSprop/sequential/encoder_conv4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*<
shared_name-+RMSprop/sequential/encoder_conv4/kernel/rms
│
?RMSprop/sequential/encoder_conv4/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv4/kernel/rms*&
_output_shapes
:0@*
dtype0
ф
)RMSprop/sequential/encoder_conv4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)RMSprop/sequential/encoder_conv4/bias/rms
Б
=RMSprop/sequential/encoder_conv4/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv4/bias/rms*
_output_shapes
:@*
dtype0
║
+RMSprop/sequential/encoder_conv5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@P*<
shared_name-+RMSprop/sequential/encoder_conv5/kernel/rms
│
?RMSprop/sequential/encoder_conv5/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv5/kernel/rms*&
_output_shapes
:@P*
dtype0
ф
)RMSprop/sequential/encoder_conv5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*:
shared_name+)RMSprop/sequential/encoder_conv5/bias/rms
Б
=RMSprop/sequential/encoder_conv5/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv5/bias/rms*
_output_shapes
:P*
dtype0
ц
#RMSprop/sequential/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђ
ђ*4
shared_name%#RMSprop/sequential/dense/kernel/rms
Ю
7RMSprop/sequential/dense/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/sequential/dense/kernel/rms* 
_output_shapes
:
ђ
ђ*
dtype0
Џ
!RMSprop/sequential/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!RMSprop/sequential/dense/bias/rms
ћ
5RMSprop/sequential/dense/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/sequential/dense/bias/rms*
_output_shapes	
:ђ*
dtype0

NoOpNoOp
Иe
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*зd
valueжdBТd B▀d
з
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
 
 
м
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer_with_weights-3
layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
R
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
у
	0decay
1learning_rate
2momentum
3rho
4iter
*rmsљ
+rmsЉ
5rmsњ
6rmsЊ
7rmsћ
8rmsЋ
9rmsќ
:rmsЌ
;rmsў
<rmsЎ
=rmsџ
>rmsЏ
?rmsю
@rmsЮ
f
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
*12
+13
 
f
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
*12
+13
Г
trainable_variables
Ametrics

Blayers
	regularization_losses
Cnon_trainable_variables
Dlayer_regularization_losses
Elayer_metrics

	variables
 
|
F_inbound_nodes

5kernel
6bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
f
K_inbound_nodes
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
f
P_inbound_nodes
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
|
U_inbound_nodes

7kernel
8bias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
f
Z_inbound_nodes
[trainable_variables
\regularization_losses
]	variables
^	keras_api
f
__inbound_nodes
`trainable_variables
aregularization_losses
b	variables
c	keras_api
|
d_inbound_nodes

9kernel
:bias
etrainable_variables
fregularization_losses
g	variables
h	keras_api
f
i_inbound_nodes
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
f
n_inbound_nodes
otrainable_variables
pregularization_losses
q	variables
r	keras_api
|
s_inbound_nodes

;kernel
<bias
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
f
x_inbound_nodes
ytrainable_variables
zregularization_losses
{	variables
|	keras_api
h
}_inbound_nodes
~trainable_variables
regularization_losses
ђ	variables
Ђ	keras_api
Ђ
ѓ_inbound_nodes

=kernel
>bias
Ѓtrainable_variables
ёregularization_losses
Ё	variables
є	keras_api
k
Є_inbound_nodes
ѕtrainable_variables
Ѕregularization_losses
і	variables
І	keras_api
k
ї_inbound_nodes
Їtrainable_variables
јregularization_losses
Ј	variables
љ	keras_api
k
Љ_inbound_nodes
њtrainable_variables
Њregularization_losses
ћ	variables
Ћ	keras_api
Ђ
ќ_inbound_nodes

?kernel
@bias
Ќtrainable_variables
ўregularization_losses
Ў	variables
џ	keras_api
V
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
 
V
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
▓
trainable_variables
Џmetrics
юlayers
regularization_losses
Юnon_trainable_variables
 ъlayer_regularization_losses
Ъlayer_metrics
 	variables
 
 
 
▓
"trainable_variables
аmetrics
Аlayers
#regularization_losses
бnon_trainable_variables
 Бlayer_regularization_losses
цlayer_metrics
$	variables
 
 
 
▓
&trainable_variables
Цmetrics
дlayers
'regularization_losses
Дnon_trainable_variables
 еlayer_regularization_losses
Еlayer_metrics
(	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
▓
,trainable_variables
фmetrics
Фlayers
-regularization_losses
гnon_trainable_variables
 Гlayer_regularization_losses
«layer_metrics
.	variables
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUErho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEsequential/encoder_conv1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential/encoder_conv1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEsequential/encoder_conv2/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential/encoder_conv2/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEsequential/encoder_conv3/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential/encoder_conv3/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEsequential/encoder_conv4/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential/encoder_conv4/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEsequential/encoder_conv5/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential/encoder_conv5/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEsequential/dense/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEsequential/dense/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE

»0
░1
*
0
1
2
3
4
5
 
 
 
 

50
61
 

50
61
▓
Gtrainable_variables
▒metrics
▓layers
Hregularization_losses
│non_trainable_variables
 ┤layer_regularization_losses
хlayer_metrics
I	variables
 
 
 
 
▓
Ltrainable_variables
Хmetrics
иlayers
Mregularization_losses
Иnon_trainable_variables
 ╣layer_regularization_losses
║layer_metrics
N	variables
 
 
 
 
▓
Qtrainable_variables
╗metrics
╝layers
Rregularization_losses
йnon_trainable_variables
 Йlayer_regularization_losses
┐layer_metrics
S	variables
 

70
81
 

70
81
▓
Vtrainable_variables
└metrics
┴layers
Wregularization_losses
┬non_trainable_variables
 ├layer_regularization_losses
─layer_metrics
X	variables
 
 
 
 
▓
[trainable_variables
┼metrics
кlayers
\regularization_losses
Кnon_trainable_variables
 ╚layer_regularization_losses
╔layer_metrics
]	variables
 
 
 
 
▓
`trainable_variables
╩metrics
╦layers
aregularization_losses
╠non_trainable_variables
 ═layer_regularization_losses
╬layer_metrics
b	variables
 

90
:1
 

90
:1
▓
etrainable_variables
¤metrics
лlayers
fregularization_losses
Лnon_trainable_variables
 мlayer_regularization_losses
Мlayer_metrics
g	variables
 
 
 
 
▓
jtrainable_variables
нmetrics
Нlayers
kregularization_losses
оnon_trainable_variables
 Оlayer_regularization_losses
пlayer_metrics
l	variables
 
 
 
 
▓
otrainable_variables
┘metrics
┌layers
pregularization_losses
█non_trainable_variables
 ▄layer_regularization_losses
Пlayer_metrics
q	variables
 

;0
<1
 

;0
<1
▓
ttrainable_variables
яmetrics
▀layers
uregularization_losses
Яnon_trainable_variables
 рlayer_regularization_losses
Рlayer_metrics
v	variables
 
 
 
 
▓
ytrainable_variables
сmetrics
Сlayers
zregularization_losses
тnon_trainable_variables
 Тlayer_regularization_losses
уlayer_metrics
{	variables
 
 
 
 
│
~trainable_variables
Уmetrics
жlayers
regularization_losses
Жnon_trainable_variables
 вlayer_regularization_losses
Вlayer_metrics
ђ	variables
 

=0
>1
 

=0
>1
х
Ѓtrainable_variables
ьmetrics
Ьlayers
ёregularization_losses
№non_trainable_variables
 ­layer_regularization_losses
ыlayer_metrics
Ё	variables
 
 
 
 
х
ѕtrainable_variables
Ыmetrics
зlayers
Ѕregularization_losses
Зnon_trainable_variables
 шlayer_regularization_losses
Шlayer_metrics
і	variables
 
 
 
 
х
Їtrainable_variables
эmetrics
Эlayers
јregularization_losses
щnon_trainable_variables
 Щlayer_regularization_losses
чlayer_metrics
Ј	variables
 
 
 
 
х
њtrainable_variables
Чmetrics
§layers
Њregularization_losses
■non_trainable_variables
  layer_regularization_losses
ђlayer_metrics
ћ	variables
 

?0
@1
 

?0
@1
х
Ќtrainable_variables
Ђmetrics
ѓlayers
ўregularization_losses
Ѓnon_trainable_variables
 ёlayer_regularization_losses
Ёlayer_metrics
Ў	variables
 
~
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

єtotal

Єcount
ѕ	variables
Ѕ	keras_api
v
іtrue_positives
Іtrue_negatives
їfalse_positives
Їfalse_negatives
ј	variables
Ј	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

є0
Є1

ѕ	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
і0
І1
ї2
Ї3

ј	variables
Ёѓ
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE+RMSprop/sequential/encoder_conv1/kernel/rmsNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE)RMSprop/sequential/encoder_conv1/bias/rmsNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE+RMSprop/sequential/encoder_conv2/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE)RMSprop/sequential/encoder_conv2/bias/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE+RMSprop/sequential/encoder_conv3/kernel/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE)RMSprop/sequential/encoder_conv3/bias/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE+RMSprop/sequential/encoder_conv4/kernel/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE)RMSprop/sequential/encoder_conv4/bias/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE+RMSprop/sequential/encoder_conv5/kernel/rmsNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
јІ
VARIABLE_VALUE)RMSprop/sequential/encoder_conv5/bias/rmsNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE#RMSprop/sequential/dense/kernel/rmsOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUE!RMSprop/sequential/dense/bias/rmsOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ј
serving_default_input_1Placeholder*1
_output_shapes
:         ђђ*
dtype0*&
shape:         ђђ
ј
serving_default_input_2Placeholder*1
_output_shapes
:         ђђ*
dtype0*&
shape:         ђђ
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2sequential/encoder_conv1/kernelsequential/encoder_conv1/biassequential/encoder_conv2/kernelsequential/encoder_conv2/biassequential/encoder_conv3/kernelsequential/encoder_conv3/biassequential/encoder_conv4/kernelsequential/encoder_conv4/biassequential/encoder_conv5/kernelsequential/encoder_conv5/biassequential/dense/kernelsequential/dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_18363
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ћ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOprho/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp3sequential/encoder_conv1/kernel/Read/ReadVariableOp1sequential/encoder_conv1/bias/Read/ReadVariableOp3sequential/encoder_conv2/kernel/Read/ReadVariableOp1sequential/encoder_conv2/bias/Read/ReadVariableOp3sequential/encoder_conv3/kernel/Read/ReadVariableOp1sequential/encoder_conv3/bias/Read/ReadVariableOp3sequential/encoder_conv4/kernel/Read/ReadVariableOp1sequential/encoder_conv4/bias/Read/ReadVariableOp3sequential/encoder_conv5/kernel/Read/ReadVariableOp1sequential/encoder_conv5/bias/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv1/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv1/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv2/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv2/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv3/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv3/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv4/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv4/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv5/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv5/bias/rms/Read/ReadVariableOp7RMSprop/sequential/dense/kernel/rms/Read/ReadVariableOp5RMSprop/sequential/dense/bias/rms/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *'
f"R 
__inference__traced_save_19379
Ѓ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasdecaylearning_ratemomentumrhoRMSprop/itersequential/encoder_conv1/kernelsequential/encoder_conv1/biassequential/encoder_conv2/kernelsequential/encoder_conv2/biassequential/encoder_conv3/kernelsequential/encoder_conv3/biassequential/encoder_conv4/kernelsequential/encoder_conv4/biassequential/encoder_conv5/kernelsequential/encoder_conv5/biassequential/dense/kernelsequential/dense/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rms+RMSprop/sequential/encoder_conv1/kernel/rms)RMSprop/sequential/encoder_conv1/bias/rms+RMSprop/sequential/encoder_conv2/kernel/rms)RMSprop/sequential/encoder_conv2/bias/rms+RMSprop/sequential/encoder_conv3/kernel/rms)RMSprop/sequential/encoder_conv3/bias/rms+RMSprop/sequential/encoder_conv4/kernel/rms)RMSprop/sequential/encoder_conv4/bias/rms+RMSprop/sequential/encoder_conv5/kernel/rms)RMSprop/sequential/encoder_conv5/bias/rms#RMSprop/sequential/dense/kernel/rms!RMSprop/sequential/dense/bias/rms*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_19506чД
зB
▓
E__inference_sequential_layer_call_and_return_conditional_losses_18872

inputs0
,encoder_conv1_conv2d_readvariableop_resource1
-encoder_conv1_biasadd_readvariableop_resource0
,encoder_conv2_conv2d_readvariableop_resource1
-encoder_conv2_biasadd_readvariableop_resource0
,encoder_conv3_conv2d_readvariableop_resource1
-encoder_conv3_biasadd_readvariableop_resource0
,encoder_conv4_conv2d_readvariableop_resource1
-encoder_conv4_biasadd_readvariableop_resource0
,encoder_conv5_conv2d_readvariableop_resource1
-encoder_conv5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityѕ┐
#encoder_conv1/Conv2D/ReadVariableOpReadVariableOp,encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#encoder_conv1/Conv2D/ReadVariableOp¤
encoder_conv1/Conv2DConv2Dinputs+encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
encoder_conv1/Conv2DХ
$encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder_conv1/BiasAdd/ReadVariableOp┬
encoder_conv1/BiasAddBiasAddencoder_conv1/Conv2D:output:0,encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
encoder_conv1/BiasAddЅ
encoder_conv1/EluEluencoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
encoder_conv1/Eluк
encoder_pool1/MaxPoolMaxPoolencoder_conv1/Elu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2
encoder_pool1/MaxPoolю
encoder_dropout1/IdentityIdentityencoder_pool1/MaxPool:output:0*
T0*/
_output_shapes
:         @@2
encoder_dropout1/Identity┐
#encoder_conv2/Conv2D/ReadVariableOpReadVariableOp,encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#encoder_conv2/Conv2D/ReadVariableOpж
encoder_conv2/Conv2DConv2D"encoder_dropout1/Identity:output:0+encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
encoder_conv2/Conv2DХ
$encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$encoder_conv2/BiasAdd/ReadVariableOp└
encoder_conv2/BiasAddBiasAddencoder_conv2/Conv2D:output:0,encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
encoder_conv2/BiasAddЄ
encoder_conv2/EluEluencoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
encoder_conv2/Eluк
encoder_pool2/MaxPoolMaxPoolencoder_conv2/Elu:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2
encoder_pool2/MaxPoolю
encoder_dropout2/IdentityIdentityencoder_pool2/MaxPool:output:0*
T0*/
_output_shapes
:            2
encoder_dropout2/Identity┐
#encoder_conv3/Conv2D/ReadVariableOpReadVariableOp,encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02%
#encoder_conv3/Conv2D/ReadVariableOpж
encoder_conv3/Conv2DConv2D"encoder_dropout2/Identity:output:0+encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2
encoder_conv3/Conv2DХ
$encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$encoder_conv3/BiasAdd/ReadVariableOp└
encoder_conv3/BiasAddBiasAddencoder_conv3/Conv2D:output:0,encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02
encoder_conv3/BiasAddЄ
encoder_conv3/EluEluencoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:           02
encoder_conv3/Eluк
encoder_pool3/MaxPoolMaxPoolencoder_conv3/Elu:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2
encoder_pool3/MaxPoolю
encoder_dropout3/IdentityIdentityencoder_pool3/MaxPool:output:0*
T0*/
_output_shapes
:         02
encoder_dropout3/Identity┐
#encoder_conv4/Conv2D/ReadVariableOpReadVariableOp,encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02%
#encoder_conv4/Conv2D/ReadVariableOpж
encoder_conv4/Conv2DConv2D"encoder_dropout3/Identity:output:0+encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
encoder_conv4/Conv2DХ
$encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$encoder_conv4/BiasAdd/ReadVariableOp└
encoder_conv4/BiasAddBiasAddencoder_conv4/Conv2D:output:0,encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
encoder_conv4/BiasAddЄ
encoder_conv4/EluEluencoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
encoder_conv4/Eluк
encoder_pool4/MaxPoolMaxPoolencoder_conv4/Elu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
encoder_pool4/MaxPoolю
encoder_dropout4/IdentityIdentityencoder_pool4/MaxPool:output:0*
T0*/
_output_shapes
:         @2
encoder_dropout4/Identity┐
#encoder_conv5/Conv2D/ReadVariableOpReadVariableOp,encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02%
#encoder_conv5/Conv2D/ReadVariableOpж
encoder_conv5/Conv2DConv2D"encoder_dropout4/Identity:output:0+encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2
encoder_conv5/Conv2DХ
$encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder_conv5/BiasAdd/ReadVariableOp└
encoder_conv5/BiasAddBiasAddencoder_conv5/Conv2D:output:0,encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2
encoder_conv5/BiasAddЄ
encoder_conv5/EluEluencoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:         P2
encoder_conv5/Eluк
encoder_pool5/MaxPoolMaxPoolencoder_conv5/Elu:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2
encoder_pool5/MaxPoolю
encoder_dropout5/IdentityIdentityencoder_pool5/MaxPool:output:0*
T0*/
_output_shapes
:         P2
encoder_dropout5/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/Constю
flatten/ReshapeReshape"encoder_dropout5/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ
2
flatten/ReshapeА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02
dense/MatMul/ReadVariableOpў
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense/Sigmoidf
IdentityIdentitydense/Sigmoid:y:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ:::::::::::::Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
љ┬
«
 __inference__wrapped_model_17353
input_1
input_2H
Dfunctional_1_sequential_encoder_conv1_conv2d_readvariableop_resourceI
Efunctional_1_sequential_encoder_conv1_biasadd_readvariableop_resourceH
Dfunctional_1_sequential_encoder_conv2_conv2d_readvariableop_resourceI
Efunctional_1_sequential_encoder_conv2_biasadd_readvariableop_resourceH
Dfunctional_1_sequential_encoder_conv3_conv2d_readvariableop_resourceI
Efunctional_1_sequential_encoder_conv3_biasadd_readvariableop_resourceH
Dfunctional_1_sequential_encoder_conv4_conv2d_readvariableop_resourceI
Efunctional_1_sequential_encoder_conv4_biasadd_readvariableop_resourceH
Dfunctional_1_sequential_encoder_conv5_conv2d_readvariableop_resourceI
Efunctional_1_sequential_encoder_conv5_biasadd_readvariableop_resource@
<functional_1_sequential_dense_matmul_readvariableop_resourceA
=functional_1_sequential_dense_biasadd_readvariableop_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource
identityѕЄ
;functional_1/sequential/encoder_conv1/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;functional_1/sequential/encoder_conv1/Conv2D/ReadVariableOpў
,functional_1/sequential/encoder_conv1/Conv2DConv2Dinput_1Cfunctional_1/sequential/encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv1/Conv2D■
<functional_1/sequential/encoder_conv1/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<functional_1/sequential/encoder_conv1/BiasAdd/ReadVariableOpб
-functional_1/sequential/encoder_conv1/BiasAddBiasAdd5functional_1/sequential/encoder_conv1/Conv2D:output:0Dfunctional_1/sequential/encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2/
-functional_1/sequential/encoder_conv1/BiasAddЛ
)functional_1/sequential/encoder_conv1/EluElu6functional_1/sequential/encoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2+
)functional_1/sequential/encoder_conv1/Eluј
-functional_1/sequential/encoder_pool1/MaxPoolMaxPool7functional_1/sequential/encoder_conv1/Elu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool1/MaxPoolС
1functional_1/sequential/encoder_dropout1/IdentityIdentity6functional_1/sequential/encoder_pool1/MaxPool:output:0*
T0*/
_output_shapes
:         @@23
1functional_1/sequential/encoder_dropout1/IdentityЄ
;functional_1/sequential/encoder_conv2/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;functional_1/sequential/encoder_conv2/Conv2D/ReadVariableOp╔
,functional_1/sequential/encoder_conv2/Conv2DConv2D:functional_1/sequential/encoder_dropout1/Identity:output:0Cfunctional_1/sequential/encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv2/Conv2D■
<functional_1/sequential/encoder_conv2/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<functional_1/sequential/encoder_conv2/BiasAdd/ReadVariableOpа
-functional_1/sequential/encoder_conv2/BiasAddBiasAdd5functional_1/sequential/encoder_conv2/Conv2D:output:0Dfunctional_1/sequential/encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2/
-functional_1/sequential/encoder_conv2/BiasAdd¤
)functional_1/sequential/encoder_conv2/EluElu6functional_1/sequential/encoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2+
)functional_1/sequential/encoder_conv2/Eluј
-functional_1/sequential/encoder_pool2/MaxPoolMaxPool7functional_1/sequential/encoder_conv2/Elu:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool2/MaxPoolС
1functional_1/sequential/encoder_dropout2/IdentityIdentity6functional_1/sequential/encoder_pool2/MaxPool:output:0*
T0*/
_output_shapes
:            23
1functional_1/sequential/encoder_dropout2/IdentityЄ
;functional_1/sequential/encoder_conv3/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;functional_1/sequential/encoder_conv3/Conv2D/ReadVariableOp╔
,functional_1/sequential/encoder_conv3/Conv2DConv2D:functional_1/sequential/encoder_dropout2/Identity:output:0Cfunctional_1/sequential/encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv3/Conv2D■
<functional_1/sequential/encoder_conv3/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02>
<functional_1/sequential/encoder_conv3/BiasAdd/ReadVariableOpа
-functional_1/sequential/encoder_conv3/BiasAddBiasAdd5functional_1/sequential/encoder_conv3/Conv2D:output:0Dfunctional_1/sequential/encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02/
-functional_1/sequential/encoder_conv3/BiasAdd¤
)functional_1/sequential/encoder_conv3/EluElu6functional_1/sequential/encoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:           02+
)functional_1/sequential/encoder_conv3/Eluј
-functional_1/sequential/encoder_pool3/MaxPoolMaxPool7functional_1/sequential/encoder_conv3/Elu:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool3/MaxPoolС
1functional_1/sequential/encoder_dropout3/IdentityIdentity6functional_1/sequential/encoder_pool3/MaxPool:output:0*
T0*/
_output_shapes
:         023
1functional_1/sequential/encoder_dropout3/IdentityЄ
;functional_1/sequential/encoder_conv4/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02=
;functional_1/sequential/encoder_conv4/Conv2D/ReadVariableOp╔
,functional_1/sequential/encoder_conv4/Conv2DConv2D:functional_1/sequential/encoder_dropout3/Identity:output:0Cfunctional_1/sequential/encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv4/Conv2D■
<functional_1/sequential/encoder_conv4/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<functional_1/sequential/encoder_conv4/BiasAdd/ReadVariableOpа
-functional_1/sequential/encoder_conv4/BiasAddBiasAdd5functional_1/sequential/encoder_conv4/Conv2D:output:0Dfunctional_1/sequential/encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2/
-functional_1/sequential/encoder_conv4/BiasAdd¤
)functional_1/sequential/encoder_conv4/EluElu6functional_1/sequential/encoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2+
)functional_1/sequential/encoder_conv4/Eluј
-functional_1/sequential/encoder_pool4/MaxPoolMaxPool7functional_1/sequential/encoder_conv4/Elu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool4/MaxPoolС
1functional_1/sequential/encoder_dropout4/IdentityIdentity6functional_1/sequential/encoder_pool4/MaxPool:output:0*
T0*/
_output_shapes
:         @23
1functional_1/sequential/encoder_dropout4/IdentityЄ
;functional_1/sequential/encoder_conv5/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02=
;functional_1/sequential/encoder_conv5/Conv2D/ReadVariableOp╔
,functional_1/sequential/encoder_conv5/Conv2DConv2D:functional_1/sequential/encoder_dropout4/Identity:output:0Cfunctional_1/sequential/encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv5/Conv2D■
<functional_1/sequential/encoder_conv5/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02>
<functional_1/sequential/encoder_conv5/BiasAdd/ReadVariableOpа
-functional_1/sequential/encoder_conv5/BiasAddBiasAdd5functional_1/sequential/encoder_conv5/Conv2D:output:0Dfunctional_1/sequential/encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2/
-functional_1/sequential/encoder_conv5/BiasAdd¤
)functional_1/sequential/encoder_conv5/EluElu6functional_1/sequential/encoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:         P2+
)functional_1/sequential/encoder_conv5/Eluј
-functional_1/sequential/encoder_pool5/MaxPoolMaxPool7functional_1/sequential/encoder_conv5/Elu:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool5/MaxPoolС
1functional_1/sequential/encoder_dropout5/IdentityIdentity6functional_1/sequential/encoder_pool5/MaxPool:output:0*
T0*/
_output_shapes
:         P23
1functional_1/sequential/encoder_dropout5/IdentityЪ
%functional_1/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%functional_1/sequential/flatten/ConstЧ
'functional_1/sequential/flatten/ReshapeReshape:functional_1/sequential/encoder_dropout5/Identity:output:0.functional_1/sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ
2)
'functional_1/sequential/flatten/Reshapeж
3functional_1/sequential/dense/MatMul/ReadVariableOpReadVariableOp<functional_1_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype025
3functional_1/sequential/dense/MatMul/ReadVariableOpЭ
$functional_1/sequential/dense/MatMulMatMul0functional_1/sequential/flatten/Reshape:output:0;functional_1/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2&
$functional_1/sequential/dense/MatMulу
4functional_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=functional_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4functional_1/sequential/dense/BiasAdd/ReadVariableOpЩ
%functional_1/sequential/dense/BiasAddBiasAdd.functional_1/sequential/dense/MatMul:product:0<functional_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2'
%functional_1/sequential/dense/BiasAdd╝
%functional_1/sequential/dense/SigmoidSigmoid.functional_1/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2'
%functional_1/sequential/dense/SigmoidІ
=functional_1/sequential/encoder_conv1/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02?
=functional_1/sequential/encoder_conv1/Conv2D_1/ReadVariableOpъ
.functional_1/sequential/encoder_conv1/Conv2D_1Conv2Dinput_2Efunctional_1/sequential/encoder_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv1/Conv2D_1ѓ
>functional_1/sequential/encoder_conv1/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>functional_1/sequential/encoder_conv1/BiasAdd_1/ReadVariableOpф
/functional_1/sequential/encoder_conv1/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv1/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ21
/functional_1/sequential/encoder_conv1/BiasAdd_1О
+functional_1/sequential/encoder_conv1/Elu_1Elu8functional_1/sequential/encoder_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:         ђђ2-
+functional_1/sequential/encoder_conv1/Elu_1ћ
/functional_1/sequential/encoder_pool1/MaxPool_1MaxPool9functional_1/sequential/encoder_conv1/Elu_1:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool1/MaxPool_1Ж
3functional_1/sequential/encoder_dropout1/Identity_1Identity8functional_1/sequential/encoder_pool1/MaxPool_1:output:0*
T0*/
_output_shapes
:         @@25
3functional_1/sequential/encoder_dropout1/Identity_1І
=functional_1/sequential/encoder_conv2/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=functional_1/sequential/encoder_conv2/Conv2D_1/ReadVariableOpЛ
.functional_1/sequential/encoder_conv2/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout1/Identity_1:output:0Efunctional_1/sequential/encoder_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv2/Conv2D_1ѓ
>functional_1/sequential/encoder_conv2/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>functional_1/sequential/encoder_conv2/BiasAdd_1/ReadVariableOpе
/functional_1/sequential/encoder_conv2/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv2/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 21
/functional_1/sequential/encoder_conv2/BiasAdd_1Н
+functional_1/sequential/encoder_conv2/Elu_1Elu8functional_1/sequential/encoder_conv2/BiasAdd_1:output:0*
T0*/
_output_shapes
:         @@ 2-
+functional_1/sequential/encoder_conv2/Elu_1ћ
/functional_1/sequential/encoder_pool2/MaxPool_1MaxPool9functional_1/sequential/encoder_conv2/Elu_1:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool2/MaxPool_1Ж
3functional_1/sequential/encoder_dropout2/Identity_1Identity8functional_1/sequential/encoder_pool2/MaxPool_1:output:0*
T0*/
_output_shapes
:            25
3functional_1/sequential/encoder_dropout2/Identity_1І
=functional_1/sequential/encoder_conv3/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02?
=functional_1/sequential/encoder_conv3/Conv2D_1/ReadVariableOpЛ
.functional_1/sequential/encoder_conv3/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout2/Identity_1:output:0Efunctional_1/sequential/encoder_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv3/Conv2D_1ѓ
>functional_1/sequential/encoder_conv3/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02@
>functional_1/sequential/encoder_conv3/BiasAdd_1/ReadVariableOpе
/functional_1/sequential/encoder_conv3/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv3/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           021
/functional_1/sequential/encoder_conv3/BiasAdd_1Н
+functional_1/sequential/encoder_conv3/Elu_1Elu8functional_1/sequential/encoder_conv3/BiasAdd_1:output:0*
T0*/
_output_shapes
:           02-
+functional_1/sequential/encoder_conv3/Elu_1ћ
/functional_1/sequential/encoder_pool3/MaxPool_1MaxPool9functional_1/sequential/encoder_conv3/Elu_1:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool3/MaxPool_1Ж
3functional_1/sequential/encoder_dropout3/Identity_1Identity8functional_1/sequential/encoder_pool3/MaxPool_1:output:0*
T0*/
_output_shapes
:         025
3functional_1/sequential/encoder_dropout3/Identity_1І
=functional_1/sequential/encoder_conv4/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02?
=functional_1/sequential/encoder_conv4/Conv2D_1/ReadVariableOpЛ
.functional_1/sequential/encoder_conv4/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout3/Identity_1:output:0Efunctional_1/sequential/encoder_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv4/Conv2D_1ѓ
>functional_1/sequential/encoder_conv4/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02@
>functional_1/sequential/encoder_conv4/BiasAdd_1/ReadVariableOpе
/functional_1/sequential/encoder_conv4/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv4/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @21
/functional_1/sequential/encoder_conv4/BiasAdd_1Н
+functional_1/sequential/encoder_conv4/Elu_1Elu8functional_1/sequential/encoder_conv4/BiasAdd_1:output:0*
T0*/
_output_shapes
:         @2-
+functional_1/sequential/encoder_conv4/Elu_1ћ
/functional_1/sequential/encoder_pool4/MaxPool_1MaxPool9functional_1/sequential/encoder_conv4/Elu_1:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool4/MaxPool_1Ж
3functional_1/sequential/encoder_dropout4/Identity_1Identity8functional_1/sequential/encoder_pool4/MaxPool_1:output:0*
T0*/
_output_shapes
:         @25
3functional_1/sequential/encoder_dropout4/Identity_1І
=functional_1/sequential/encoder_conv5/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02?
=functional_1/sequential/encoder_conv5/Conv2D_1/ReadVariableOpЛ
.functional_1/sequential/encoder_conv5/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout4/Identity_1:output:0Efunctional_1/sequential/encoder_conv5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv5/Conv2D_1ѓ
>functional_1/sequential/encoder_conv5/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02@
>functional_1/sequential/encoder_conv5/BiasAdd_1/ReadVariableOpе
/functional_1/sequential/encoder_conv5/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv5/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P21
/functional_1/sequential/encoder_conv5/BiasAdd_1Н
+functional_1/sequential/encoder_conv5/Elu_1Elu8functional_1/sequential/encoder_conv5/BiasAdd_1:output:0*
T0*/
_output_shapes
:         P2-
+functional_1/sequential/encoder_conv5/Elu_1ћ
/functional_1/sequential/encoder_pool5/MaxPool_1MaxPool9functional_1/sequential/encoder_conv5/Elu_1:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool5/MaxPool_1Ж
3functional_1/sequential/encoder_dropout5/Identity_1Identity8functional_1/sequential/encoder_pool5/MaxPool_1:output:0*
T0*/
_output_shapes
:         P25
3functional_1/sequential/encoder_dropout5/Identity_1Б
'functional_1/sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'functional_1/sequential/flatten/Const_1ё
)functional_1/sequential/flatten/Reshape_1Reshape<functional_1/sequential/encoder_dropout5/Identity_1:output:00functional_1/sequential/flatten/Const_1:output:0*
T0*(
_output_shapes
:         ђ
2+
)functional_1/sequential/flatten/Reshape_1ь
5functional_1/sequential/dense/MatMul_1/ReadVariableOpReadVariableOp<functional_1_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype027
5functional_1/sequential/dense/MatMul_1/ReadVariableOpђ
&functional_1/sequential/dense/MatMul_1MatMul2functional_1/sequential/flatten/Reshape_1:output:0=functional_1/sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&functional_1/sequential/dense/MatMul_1в
6functional_1/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp=functional_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6functional_1/sequential/dense/BiasAdd_1/ReadVariableOpѓ
'functional_1/sequential/dense/BiasAdd_1BiasAdd0functional_1/sequential/dense/MatMul_1:product:0>functional_1/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2)
'functional_1/sequential/dense/BiasAdd_1┬
'functional_1/sequential/dense/Sigmoid_1Sigmoid0functional_1/sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:         ђ2)
'functional_1/sequential/dense/Sigmoid_1╚
functional_1/subtract/subSub)functional_1/sequential/dense/Sigmoid:y:0+functional_1/sequential/dense/Sigmoid_1:y:0*
T0*(
_output_shapes
:         ђ2
functional_1/subtract/subг
 functional_1/tf_op_layer_Abs/AbsAbsfunctional_1/subtract/sub:z:0*
T0*
_cloned(*(
_output_shapes
:         ђ2"
 functional_1/tf_op_layer_Abs/Abs═
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpл
functional_1/dense_1/MatMulMatMul$functional_1/tf_op_layer_Abs/Abs:y:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_1/dense_1/MatMul╦
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpН
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
functional_1/dense_1/BiasAddа
functional_1/dense_1/SigmoidSigmoid%functional_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
functional_1/dense_1/Sigmoidt
IdentityIdentity functional_1/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ:::::::::::::::Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:         ђђ
!
_user_specified_name	input_2
╔
j
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17631

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_19009

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @@:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
ЯW
в
__inference__traced_save_19379
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop"
savev2_rho_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	>
:savev2_sequential_encoder_conv1_kernel_read_readvariableop<
8savev2_sequential_encoder_conv1_bias_read_readvariableop>
:savev2_sequential_encoder_conv2_kernel_read_readvariableop<
8savev2_sequential_encoder_conv2_bias_read_readvariableop>
:savev2_sequential_encoder_conv3_kernel_read_readvariableop<
8savev2_sequential_encoder_conv3_bias_read_readvariableop>
:savev2_sequential_encoder_conv4_kernel_read_readvariableop<
8savev2_sequential_encoder_conv4_bias_read_readvariableop>
:savev2_sequential_encoder_conv5_kernel_read_readvariableop<
8savev2_sequential_encoder_conv5_bias_read_readvariableop6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableopJ
Fsavev2_rmsprop_sequential_encoder_conv1_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_sequential_encoder_conv1_bias_rms_read_readvariableopJ
Fsavev2_rmsprop_sequential_encoder_conv2_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_sequential_encoder_conv2_bias_rms_read_readvariableopJ
Fsavev2_rmsprop_sequential_encoder_conv3_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_sequential_encoder_conv3_bias_rms_read_readvariableopJ
Fsavev2_rmsprop_sequential_encoder_conv4_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_sequential_encoder_conv4_bias_rms_read_readvariableopJ
Fsavev2_rmsprop_sequential_encoder_conv5_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_sequential_encoder_conv5_bias_rms_read_readvariableopB
>savev2_rmsprop_sequential_dense_kernel_rms_read_readvariableop@
<savev2_rmsprop_sequential_dense_bias_rms_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_dc1852cf85854899a314aec5595c5064/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameБ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*х
valueФBе(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesп
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableopsavev2_rho_read_readvariableop'savev2_rmsprop_iter_read_readvariableop:savev2_sequential_encoder_conv1_kernel_read_readvariableop8savev2_sequential_encoder_conv1_bias_read_readvariableop:savev2_sequential_encoder_conv2_kernel_read_readvariableop8savev2_sequential_encoder_conv2_bias_read_readvariableop:savev2_sequential_encoder_conv3_kernel_read_readvariableop8savev2_sequential_encoder_conv3_bias_read_readvariableop:savev2_sequential_encoder_conv4_kernel_read_readvariableop8savev2_sequential_encoder_conv4_bias_read_readvariableop:savev2_sequential_encoder_conv5_kernel_read_readvariableop8savev2_sequential_encoder_conv5_bias_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv1_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv1_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv2_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv2_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv3_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv3_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv4_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv4_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv5_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv5_bias_rms_read_readvariableop>savev2_rmsprop_sequential_dense_kernel_rms_read_readvariableop<savev2_rmsprop_sequential_dense_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ч
_input_shapesж
Т: :	ђ:: : : : : ::: : : 0:0:0@:@:@P:P:
ђ
ђ:ђ: : :╚:╚:╚:╚:	ђ:::: : : 0:0:0@:@:@P:P:
ђ
ђ:ђ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@:,(
&
_output_shapes
:@P: 

_output_shapes
:P:&"
 
_output_shapes
:
ђ
ђ:!

_output_shapes	
:ђ:

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:%!

_output_shapes
:	ђ: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: 0: !

_output_shapes
:0:,"(
&
_output_shapes
:0@: #

_output_shapes
:@:,$(
&
_output_shapes
:@P: %

_output_shapes
:P:&&"
 
_output_shapes
:
ђ
ђ:!'

_output_shapes	
:ђ:(

_output_shapes
: 
└
L
0__inference_encoder_dropout2_layer_call_fn_19066

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_175202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:            2

Identity"
identityIdentity:output:0*.
_input_shapes
:            :W S
/
_output_shapes
:            
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_19051

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:            2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:            *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:            2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:            2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:            2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:            2

Identity"
identityIdentity:output:0*.
_input_shapes
:            :W S
/
_output_shapes
:            
 
_user_specified_nameinputs
§
d
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_17407

inputs
identityг
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ
к
,__inference_functional_1_layer_call_fn_18235
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_182042
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:         ђђ
!
_user_specified_name	input_2
└
L
0__inference_encoder_dropout3_layer_call_fn_19113

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_175782
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         02

Identity"
identityIdentity:output:0*.
_input_shapes
:         0:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
ѓE
У
E__inference_sequential_layer_call_and_return_conditional_losses_17916

inputs
encoder_conv1_17874
encoder_conv1_17876
encoder_conv2_17881
encoder_conv2_17883
encoder_conv3_17888
encoder_conv3_17890
encoder_conv4_17895
encoder_conv4_17897
encoder_conv5_17902
encoder_conv5_17904
dense_17910
dense_17912
identityѕбdense/StatefulPartitionedCallб%encoder_conv1/StatefulPartitionedCallб%encoder_conv2/StatefulPartitionedCallб%encoder_conv3/StatefulPartitionedCallб%encoder_conv4/StatefulPartitionedCallб%encoder_conv5/StatefulPartitionedCall┤
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_conv1_17874encoder_conv1_17876*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_174282'
%encoder_conv1/StatefulPartitionedCallњ
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_173592
encoder_pool1/PartitionedCallЊ
 encoder_dropout1/PartitionedCallPartitionedCall&encoder_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_174622"
 encoder_dropout1/PartitionedCallН
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout1/PartitionedCall:output:0encoder_conv2_17881encoder_conv2_17883*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_174862'
%encoder_conv2/StatefulPartitionedCallњ
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_173712
encoder_pool2/PartitionedCallЊ
 encoder_dropout2/PartitionedCallPartitionedCall&encoder_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_175202"
 encoder_dropout2/PartitionedCallН
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout2/PartitionedCall:output:0encoder_conv3_17888encoder_conv3_17890*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_175442'
%encoder_conv3/StatefulPartitionedCallњ
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_173832
encoder_pool3/PartitionedCallЊ
 encoder_dropout3/PartitionedCallPartitionedCall&encoder_pool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_175782"
 encoder_dropout3/PartitionedCallН
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout3/PartitionedCall:output:0encoder_conv4_17895encoder_conv4_17897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_176022'
%encoder_conv4/StatefulPartitionedCallњ
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_173952
encoder_pool4/PartitionedCallЊ
 encoder_dropout4/PartitionedCallPartitionedCall&encoder_pool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_176362"
 encoder_dropout4/PartitionedCallН
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout4/PartitionedCall:output:0encoder_conv5_17902encoder_conv5_17904*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_176602'
%encoder_conv5/StatefulPartitionedCallњ
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_174072
encoder_pool5/PartitionedCallЊ
 encoder_dropout5/PartitionedCallPartitionedCall&encoder_pool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_176942"
 encoder_dropout5/PartitionedCallЗ
flatten/PartitionedCallPartitionedCall)encoder_dropout5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_177132
flatten/PartitionedCallЮ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_17910dense_17912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_177322
dense/StatefulPartitionedCallс
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%encoder_conv1/StatefulPartitionedCall%encoder_conv1/StatefulPartitionedCall2N
%encoder_conv2/StatefulPartitionedCall%encoder_conv2/StatefulPartitionedCall2N
%encoder_conv3/StatefulPartitionedCall%encoder_conv3/StatefulPartitionedCall2N
%encoder_conv4/StatefulPartitionedCall%encoder_conv4/StatefulPartitionedCall2N
%encoder_conv5/StatefulPartitionedCall%encoder_conv5/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_19192

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         P2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         P2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
г
ф
B__inference_dense_1_layer_call_and_return_conditional_losses_18963

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ѕ	
░
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_17486

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@:::W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
з
љ
G__inference_functional_1_layer_call_and_return_conditional_losses_18100
input_1
input_2
sequential_18006
sequential_18008
sequential_18010
sequential_18012
sequential_18014
sequential_18016
sequential_18018
sequential_18020
sequential_18022
sequential_18024
sequential_18026
sequential_18028
dense_1_18094
dense_1_18096
identityѕбdense_1/StatefulPartitionedCallб"sequential/StatefulPartitionedCallб$sequential/StatefulPartitionedCall_1т
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_18006sequential_18008sequential_18010sequential_18012sequential_18014sequential_18016sequential_18018sequential_18020sequential_18022sequential_18024sequential_18026sequential_18028*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178422$
"sequential/StatefulPartitionedCallж
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_18006sequential_18008sequential_18010sequential_18012sequential_18014sequential_18016sequential_18018sequential_18020sequential_18022sequential_18024sequential_18026sequential_18028*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178422&
$sequential/StatefulPartitionedCall_1Е
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_180502
subtract/PartitionedCallё
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_180642!
tf_op_layer_Abs/PartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_18094dense_1_18096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_180832!
dense_1/StatefulPartitionedCallЖ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:         ђђ
!
_user_specified_name	input_2
Ќ
╚
,__inference_functional_1_layer_call_fn_18687
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_182042
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1
Ё
ѓ
-__inference_encoder_conv5_layer_call_fn_19180

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_176602
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
я	
Б
*__inference_sequential_layer_call_fn_17869
encoder_conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallencoder_conv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178422
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:         ђђ
-
_user_specified_nameencoder_conv1_input
й
f
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_18947

inputs
identity[
AbsAbsinputs*
T0*
_cloned(*(
_output_shapes
:         ђ2
Abs\
IdentityIdentityAbs:y:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
и	
ќ
*__inference_sequential_layer_call_fn_18930

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_179162
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ы
љ
G__inference_functional_1_layer_call_and_return_conditional_losses_18288

inputs
inputs_1
sequential_18242
sequential_18244
sequential_18246
sequential_18248
sequential_18250
sequential_18252
sequential_18254
sequential_18256
sequential_18258
sequential_18260
sequential_18262
sequential_18264
dense_1_18282
dense_1_18284
identityѕбdense_1/StatefulPartitionedCallб"sequential/StatefulPartitionedCallб$sequential/StatefulPartitionedCall_1С
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_18242sequential_18244sequential_18246sequential_18248sequential_18250sequential_18252sequential_18254sequential_18256sequential_18258sequential_18260sequential_18262sequential_18264*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_179162$
"sequential/StatefulPartitionedCallЖ
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_18242sequential_18244sequential_18246sequential_18248sequential_18250sequential_18252sequential_18254sequential_18256sequential_18258sequential_18260sequential_18262sequential_18264*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_179162&
$sequential/StatefulPartitionedCall_1Е
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_180502
subtract/PartitionedCallё
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_180642!
tf_op_layer_Abs/PartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_18282dense_1_18284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_180832!
dense_1/StatefulPartitionedCallЖ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
╠
i
0__inference_encoder_dropout3_layer_call_fn_19108

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_175732
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         02

Identity"
identityIdentity:output:0*.
_input_shapes
:         022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
Џ
T
(__inference_subtract_layer_call_fn_18942
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_180502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ђ:         ђ:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/1
Ь
i
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_17462

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @@:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
§
d
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_17383

inputs
identityг
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ы
љ
G__inference_functional_1_layer_call_and_return_conditional_losses_18204

inputs
inputs_1
sequential_18158
sequential_18160
sequential_18162
sequential_18164
sequential_18166
sequential_18168
sequential_18170
sequential_18172
sequential_18174
sequential_18176
sequential_18178
sequential_18180
dense_1_18198
dense_1_18200
identityѕбdense_1/StatefulPartitionedCallб"sequential/StatefulPartitionedCallб$sequential/StatefulPartitionedCall_1С
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_18158sequential_18160sequential_18162sequential_18164sequential_18166sequential_18168sequential_18170sequential_18172sequential_18174sequential_18176sequential_18178sequential_18180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178422$
"sequential/StatefulPartitionedCallЖ
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_18158sequential_18160sequential_18162sequential_18164sequential_18166sequential_18168sequential_18170sequential_18172sequential_18174sequential_18176sequential_18178sequential_18180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178422&
$sequential/StatefulPartitionedCall_1Е
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_180502
subtract/PartitionedCallё
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_180642!
tf_op_layer_Abs/PartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_18198dense_1_18200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_180832!
dense_1/StatefulPartitionedCallЖ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
§
d
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_17395

inputs
identityг
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
й
f
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_18064

inputs
identity[
AbsAbsinputs*
T0*
_cloned(*(
_output_shapes
:         ђ2
Abs\
IdentityIdentityAbs:y:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ћ	
░
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_18983

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
Eluo
IdentityIdentityElu:activations:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ђђ:::Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Д
I
-__inference_encoder_pool3_layer_call_fn_17389

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_173832
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
а
C
'__inference_flatten_layer_call_fn_19218

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_177132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17694

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         P2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
б
K
/__inference_tf_op_layer_Abs_layer_call_fn_18952

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_180642
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17636

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Об
А
G__inference_functional_1_layer_call_and_return_conditional_losses_18653
inputs_0
inputs_1;
7sequential_encoder_conv1_conv2d_readvariableop_resource<
8sequential_encoder_conv1_biasadd_readvariableop_resource;
7sequential_encoder_conv2_conv2d_readvariableop_resource<
8sequential_encoder_conv2_biasadd_readvariableop_resource;
7sequential_encoder_conv3_conv2d_readvariableop_resource<
8sequential_encoder_conv3_biasadd_readvariableop_resource;
7sequential_encoder_conv4_conv2d_readvariableop_resource<
8sequential_encoder_conv4_biasadd_readvariableop_resource;
7sequential_encoder_conv5_conv2d_readvariableop_resource<
8sequential_encoder_conv5_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕЯ
.sequential/encoder_conv1/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.sequential/encoder_conv1/Conv2D/ReadVariableOpЫ
sequential/encoder_conv1/Conv2DConv2Dinputs_06sequential/encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2!
sequential/encoder_conv1/Conv2DО
/sequential/encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/encoder_conv1/BiasAdd/ReadVariableOpЬ
 sequential/encoder_conv1/BiasAddBiasAdd(sequential/encoder_conv1/Conv2D:output:07sequential/encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2"
 sequential/encoder_conv1/BiasAddф
sequential/encoder_conv1/EluElu)sequential/encoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
sequential/encoder_conv1/Eluу
 sequential/encoder_pool1/MaxPoolMaxPool*sequential/encoder_conv1/Elu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool1/MaxPoolй
$sequential/encoder_dropout1/IdentityIdentity)sequential/encoder_pool1/MaxPool:output:0*
T0*/
_output_shapes
:         @@2&
$sequential/encoder_dropout1/IdentityЯ
.sequential/encoder_conv2/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential/encoder_conv2/Conv2D/ReadVariableOpЋ
sequential/encoder_conv2/Conv2DConv2D-sequential/encoder_dropout1/Identity:output:06sequential/encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2!
sequential/encoder_conv2/Conv2DО
/sequential/encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/encoder_conv2/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv2/BiasAddBiasAdd(sequential/encoder_conv2/Conv2D:output:07sequential/encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2"
 sequential/encoder_conv2/BiasAddе
sequential/encoder_conv2/EluElu)sequential/encoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
sequential/encoder_conv2/Eluу
 sequential/encoder_pool2/MaxPoolMaxPool*sequential/encoder_conv2/Elu:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool2/MaxPoolй
$sequential/encoder_dropout2/IdentityIdentity)sequential/encoder_pool2/MaxPool:output:0*
T0*/
_output_shapes
:            2&
$sequential/encoder_dropout2/IdentityЯ
.sequential/encoder_conv3/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype020
.sequential/encoder_conv3/Conv2D/ReadVariableOpЋ
sequential/encoder_conv3/Conv2DConv2D-sequential/encoder_dropout2/Identity:output:06sequential/encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2!
sequential/encoder_conv3/Conv2DО
/sequential/encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/encoder_conv3/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv3/BiasAddBiasAdd(sequential/encoder_conv3/Conv2D:output:07sequential/encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02"
 sequential/encoder_conv3/BiasAddе
sequential/encoder_conv3/EluElu)sequential/encoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:           02
sequential/encoder_conv3/Eluу
 sequential/encoder_pool3/MaxPoolMaxPool*sequential/encoder_conv3/Elu:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool3/MaxPoolй
$sequential/encoder_dropout3/IdentityIdentity)sequential/encoder_pool3/MaxPool:output:0*
T0*/
_output_shapes
:         02&
$sequential/encoder_dropout3/IdentityЯ
.sequential/encoder_conv4/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype020
.sequential/encoder_conv4/Conv2D/ReadVariableOpЋ
sequential/encoder_conv4/Conv2DConv2D-sequential/encoder_dropout3/Identity:output:06sequential/encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2!
sequential/encoder_conv4/Conv2DО
/sequential/encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/encoder_conv4/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv4/BiasAddBiasAdd(sequential/encoder_conv4/Conv2D:output:07sequential/encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2"
 sequential/encoder_conv4/BiasAddе
sequential/encoder_conv4/EluElu)sequential/encoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
sequential/encoder_conv4/Eluу
 sequential/encoder_pool4/MaxPoolMaxPool*sequential/encoder_conv4/Elu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool4/MaxPoolй
$sequential/encoder_dropout4/IdentityIdentity)sequential/encoder_pool4/MaxPool:output:0*
T0*/
_output_shapes
:         @2&
$sequential/encoder_dropout4/IdentityЯ
.sequential/encoder_conv5/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype020
.sequential/encoder_conv5/Conv2D/ReadVariableOpЋ
sequential/encoder_conv5/Conv2DConv2D-sequential/encoder_dropout4/Identity:output:06sequential/encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2!
sequential/encoder_conv5/Conv2DО
/sequential/encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype021
/sequential/encoder_conv5/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv5/BiasAddBiasAdd(sequential/encoder_conv5/Conv2D:output:07sequential/encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2"
 sequential/encoder_conv5/BiasAddе
sequential/encoder_conv5/EluElu)sequential/encoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:         P2
sequential/encoder_conv5/Eluу
 sequential/encoder_pool5/MaxPoolMaxPool*sequential/encoder_conv5/Elu:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool5/MaxPoolй
$sequential/encoder_dropout5/IdentityIdentity)sequential/encoder_pool5/MaxPool:output:0*
T0*/
_output_shapes
:         P2&
$sequential/encoder_dropout5/IdentityЁ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential/flatten/Const╚
sequential/flatten/ReshapeReshape-sequential/encoder_dropout5/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ
2
sequential/flatten/Reshape┬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02(
&sequential/dense/MatMul/ReadVariableOp─
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/MatMul└
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpк
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/BiasAddЋ
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/SigmoidС
0sequential/encoder_conv1/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0sequential/encoder_conv1/Conv2D_1/ReadVariableOpЭ
!sequential/encoder_conv1/Conv2D_1Conv2Dinputs_18sequential/encoder_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2#
!sequential/encoder_conv1/Conv2D_1█
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOpШ
"sequential/encoder_conv1/BiasAdd_1BiasAdd*sequential/encoder_conv1/Conv2D_1:output:09sequential/encoder_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2$
"sequential/encoder_conv1/BiasAdd_1░
sequential/encoder_conv1/Elu_1Elu+sequential/encoder_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:         ђђ2 
sequential/encoder_conv1/Elu_1ь
"sequential/encoder_pool1/MaxPool_1MaxPool,sequential/encoder_conv1/Elu_1:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool1/MaxPool_1├
&sequential/encoder_dropout1/Identity_1Identity+sequential/encoder_pool1/MaxPool_1:output:0*
T0*/
_output_shapes
:         @@2(
&sequential/encoder_dropout1/Identity_1С
0sequential/encoder_conv2/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0sequential/encoder_conv2/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv2/Conv2D_1Conv2D/sequential/encoder_dropout1/Identity_1:output:08sequential/encoder_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2#
!sequential/encoder_conv2/Conv2D_1█
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv2/BiasAdd_1BiasAdd*sequential/encoder_conv2/Conv2D_1:output:09sequential/encoder_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2$
"sequential/encoder_conv2/BiasAdd_1«
sequential/encoder_conv2/Elu_1Elu+sequential/encoder_conv2/BiasAdd_1:output:0*
T0*/
_output_shapes
:         @@ 2 
sequential/encoder_conv2/Elu_1ь
"sequential/encoder_pool2/MaxPool_1MaxPool,sequential/encoder_conv2/Elu_1:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool2/MaxPool_1├
&sequential/encoder_dropout2/Identity_1Identity+sequential/encoder_pool2/MaxPool_1:output:0*
T0*/
_output_shapes
:            2(
&sequential/encoder_dropout2/Identity_1С
0sequential/encoder_conv3/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype022
0sequential/encoder_conv3/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv3/Conv2D_1Conv2D/sequential/encoder_dropout2/Identity_1:output:08sequential/encoder_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2#
!sequential/encoder_conv3/Conv2D_1█
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv3/BiasAdd_1BiasAdd*sequential/encoder_conv3/Conv2D_1:output:09sequential/encoder_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02$
"sequential/encoder_conv3/BiasAdd_1«
sequential/encoder_conv3/Elu_1Elu+sequential/encoder_conv3/BiasAdd_1:output:0*
T0*/
_output_shapes
:           02 
sequential/encoder_conv3/Elu_1ь
"sequential/encoder_pool3/MaxPool_1MaxPool,sequential/encoder_conv3/Elu_1:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool3/MaxPool_1├
&sequential/encoder_dropout3/Identity_1Identity+sequential/encoder_pool3/MaxPool_1:output:0*
T0*/
_output_shapes
:         02(
&sequential/encoder_dropout3/Identity_1С
0sequential/encoder_conv4/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype022
0sequential/encoder_conv4/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv4/Conv2D_1Conv2D/sequential/encoder_dropout3/Identity_1:output:08sequential/encoder_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2#
!sequential/encoder_conv4/Conv2D_1█
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv4/BiasAdd_1BiasAdd*sequential/encoder_conv4/Conv2D_1:output:09sequential/encoder_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2$
"sequential/encoder_conv4/BiasAdd_1«
sequential/encoder_conv4/Elu_1Elu+sequential/encoder_conv4/BiasAdd_1:output:0*
T0*/
_output_shapes
:         @2 
sequential/encoder_conv4/Elu_1ь
"sequential/encoder_pool4/MaxPool_1MaxPool,sequential/encoder_conv4/Elu_1:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool4/MaxPool_1├
&sequential/encoder_dropout4/Identity_1Identity+sequential/encoder_pool4/MaxPool_1:output:0*
T0*/
_output_shapes
:         @2(
&sequential/encoder_dropout4/Identity_1С
0sequential/encoder_conv5/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype022
0sequential/encoder_conv5/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv5/Conv2D_1Conv2D/sequential/encoder_dropout4/Identity_1:output:08sequential/encoder_conv5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2#
!sequential/encoder_conv5/Conv2D_1█
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype023
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv5/BiasAdd_1BiasAdd*sequential/encoder_conv5/Conv2D_1:output:09sequential/encoder_conv5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2$
"sequential/encoder_conv5/BiasAdd_1«
sequential/encoder_conv5/Elu_1Elu+sequential/encoder_conv5/BiasAdd_1:output:0*
T0*/
_output_shapes
:         P2 
sequential/encoder_conv5/Elu_1ь
"sequential/encoder_pool5/MaxPool_1MaxPool,sequential/encoder_conv5/Elu_1:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool5/MaxPool_1├
&sequential/encoder_dropout5/Identity_1Identity+sequential/encoder_pool5/MaxPool_1:output:0*
T0*/
_output_shapes
:         P2(
&sequential/encoder_dropout5/Identity_1Ѕ
sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
sequential/flatten/Const_1л
sequential/flatten/Reshape_1Reshape/sequential/encoder_dropout5/Identity_1:output:0#sequential/flatten/Const_1:output:0*
T0*(
_output_shapes
:         ђ
2
sequential/flatten/Reshape_1к
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02*
(sequential/dense/MatMul_1/ReadVariableOp╠
sequential/dense/MatMul_1MatMul%sequential/flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/MatMul_1─
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)sequential/dense/BiasAdd_1/ReadVariableOp╬
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/BiasAdd_1Џ
sequential/dense/Sigmoid_1Sigmoid#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/Sigmoid_1ћ
subtract/subSubsequential/dense/Sigmoid:y:0sequential/dense/Sigmoid_1:y:0*
T0*(
_output_shapes
:         ђ2
subtract/subЁ
tf_op_layer_Abs/AbsAbssubtract/sub:z:0*
T0*
_cloned(*(
_output_shapes
:         ђ2
tf_op_layer_Abs/Absд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_1/MatMul/ReadVariableOpю
dense_1/MatMulMatMultf_op_layer_Abs/Abs:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/Sigmoidg
IdentityIdentitydense_1/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ:::::::::::::::[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1
░
е
@__inference_dense_layer_call_and_return_conditional_losses_17732

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ
:::P L
(
_output_shapes
:         ђ

 
_user_specified_nameinputs
г
ф
B__inference_dense_1_layer_call_and_return_conditional_losses_18083

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_19145

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ѕ	
░
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_19171

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:         P2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @:::W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
фN
┐
E__inference_sequential_layer_call_and_return_conditional_losses_17842

inputs
encoder_conv1_17800
encoder_conv1_17802
encoder_conv2_17807
encoder_conv2_17809
encoder_conv3_17814
encoder_conv3_17816
encoder_conv4_17821
encoder_conv4_17823
encoder_conv5_17828
encoder_conv5_17830
dense_17836
dense_17838
identityѕбdense/StatefulPartitionedCallб%encoder_conv1/StatefulPartitionedCallб%encoder_conv2/StatefulPartitionedCallб%encoder_conv3/StatefulPartitionedCallб%encoder_conv4/StatefulPartitionedCallб%encoder_conv5/StatefulPartitionedCallб(encoder_dropout1/StatefulPartitionedCallб(encoder_dropout2/StatefulPartitionedCallб(encoder_dropout3/StatefulPartitionedCallб(encoder_dropout4/StatefulPartitionedCallб(encoder_dropout5/StatefulPartitionedCall┤
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_conv1_17800encoder_conv1_17802*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_174282'
%encoder_conv1/StatefulPartitionedCallњ
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_173592
encoder_pool1/PartitionedCallФ
(encoder_dropout1/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_174572*
(encoder_dropout1/StatefulPartitionedCallП
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout1/StatefulPartitionedCall:output:0encoder_conv2_17807encoder_conv2_17809*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_174862'
%encoder_conv2/StatefulPartitionedCallњ
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_173712
encoder_pool2/PartitionedCallо
(encoder_dropout2/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool2/PartitionedCall:output:0)^encoder_dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_175152*
(encoder_dropout2/StatefulPartitionedCallП
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout2/StatefulPartitionedCall:output:0encoder_conv3_17814encoder_conv3_17816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_175442'
%encoder_conv3/StatefulPartitionedCallњ
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_173832
encoder_pool3/PartitionedCallо
(encoder_dropout3/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool3/PartitionedCall:output:0)^encoder_dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_175732*
(encoder_dropout3/StatefulPartitionedCallП
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout3/StatefulPartitionedCall:output:0encoder_conv4_17821encoder_conv4_17823*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_176022'
%encoder_conv4/StatefulPartitionedCallњ
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_173952
encoder_pool4/PartitionedCallо
(encoder_dropout4/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool4/PartitionedCall:output:0)^encoder_dropout3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_176312*
(encoder_dropout4/StatefulPartitionedCallП
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout4/StatefulPartitionedCall:output:0encoder_conv5_17828encoder_conv5_17830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_176602'
%encoder_conv5/StatefulPartitionedCallњ
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_174072
encoder_pool5/PartitionedCallо
(encoder_dropout5/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool5/PartitionedCall:output:0)^encoder_dropout4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_176892*
(encoder_dropout5/StatefulPartitionedCallЧ
flatten/PartitionedCallPartitionedCall1encoder_dropout5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_177132
flatten/PartitionedCallЮ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_17836dense_17838*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_177322
dense/StatefulPartitionedCall║
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall)^encoder_dropout1/StatefulPartitionedCall)^encoder_dropout2/StatefulPartitionedCall)^encoder_dropout3/StatefulPartitionedCall)^encoder_dropout4/StatefulPartitionedCall)^encoder_dropout5/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%encoder_conv1/StatefulPartitionedCall%encoder_conv1/StatefulPartitionedCall2N
%encoder_conv2/StatefulPartitionedCall%encoder_conv2/StatefulPartitionedCall2N
%encoder_conv3/StatefulPartitionedCall%encoder_conv3/StatefulPartitionedCall2N
%encoder_conv4/StatefulPartitionedCall%encoder_conv4/StatefulPartitionedCall2N
%encoder_conv5/StatefulPartitionedCall%encoder_conv5/StatefulPartitionedCall2T
(encoder_dropout1/StatefulPartitionedCall(encoder_dropout1/StatefulPartitionedCall2T
(encoder_dropout2/StatefulPartitionedCall(encoder_dropout2/StatefulPartitionedCall2T
(encoder_dropout3/StatefulPartitionedCall(encoder_dropout3/StatefulPartitionedCall2T
(encoder_dropout4/StatefulPartitionedCall(encoder_dropout4/StatefulPartitionedCall2T
(encoder_dropout5/StatefulPartitionedCall(encoder_dropout5/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ё
ѓ
-__inference_encoder_conv2_layer_call_fn_19039

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_174862
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
└
L
0__inference_encoder_dropout4_layer_call_fn_19160

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_176362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17573

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         02
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         02
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         02

Identity"
identityIdentity:output:0*.
_input_shapes
:         0:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
п
z
%__inference_dense_layer_call_fn_19238

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_177322
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ
::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ

 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_19004

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╠
i
0__inference_encoder_dropout4_layer_call_fn_19155

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_176312
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ѕ	
░
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_17660

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:         P2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @:::W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ц
m
C__inference_subtract_layer_call_and_return_conditional_losses_18050

inputs
inputs_1
identityV
subSubinputsinputs_1*
T0*(
_output_shapes
:         ђ2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ђ:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
░
е
@__inference_dense_layer_call_and_return_conditional_losses_19229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ
:::P L
(
_output_shapes
:         ђ

 
_user_specified_nameinputs
└
L
0__inference_encoder_dropout5_layer_call_fn_19207

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_176942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
└
L
0__inference_encoder_dropout1_layer_call_fn_19019

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_174622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
р

й
#__inference_signature_wrapper_18363
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_173532
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:         ђђ
!
_user_specified_name	input_2
Ѕ	
░
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_19077

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:           02
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:           02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:            :::W S
/
_output_shapes
:            
 
_user_specified_nameinputs
╠
i
0__inference_encoder_dropout2_layer_call_fn_19061

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_175152
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:            2

Identity"
identityIdentity:output:0*.
_input_shapes
:            22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:            
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_17520

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:            2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:            2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:            :W S
/
_output_shapes
:            
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_19103

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         02

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         0:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_19197

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         P2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
ЕE
ш
E__inference_sequential_layer_call_and_return_conditional_losses_17794
encoder_conv1_input
encoder_conv1_17752
encoder_conv1_17754
encoder_conv2_17759
encoder_conv2_17761
encoder_conv3_17766
encoder_conv3_17768
encoder_conv4_17773
encoder_conv4_17775
encoder_conv5_17780
encoder_conv5_17782
dense_17788
dense_17790
identityѕбdense/StatefulPartitionedCallб%encoder_conv1/StatefulPartitionedCallб%encoder_conv2/StatefulPartitionedCallб%encoder_conv3/StatefulPartitionedCallб%encoder_conv4/StatefulPartitionedCallб%encoder_conv5/StatefulPartitionedCall┴
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCallencoder_conv1_inputencoder_conv1_17752encoder_conv1_17754*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_174282'
%encoder_conv1/StatefulPartitionedCallњ
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_173592
encoder_pool1/PartitionedCallЊ
 encoder_dropout1/PartitionedCallPartitionedCall&encoder_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_174622"
 encoder_dropout1/PartitionedCallН
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout1/PartitionedCall:output:0encoder_conv2_17759encoder_conv2_17761*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_174862'
%encoder_conv2/StatefulPartitionedCallњ
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_173712
encoder_pool2/PartitionedCallЊ
 encoder_dropout2/PartitionedCallPartitionedCall&encoder_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_175202"
 encoder_dropout2/PartitionedCallН
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout2/PartitionedCall:output:0encoder_conv3_17766encoder_conv3_17768*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_175442'
%encoder_conv3/StatefulPartitionedCallњ
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_173832
encoder_pool3/PartitionedCallЊ
 encoder_dropout3/PartitionedCallPartitionedCall&encoder_pool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_175782"
 encoder_dropout3/PartitionedCallН
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout3/PartitionedCall:output:0encoder_conv4_17773encoder_conv4_17775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_176022'
%encoder_conv4/StatefulPartitionedCallњ
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_173952
encoder_pool4/PartitionedCallЊ
 encoder_dropout4/PartitionedCallPartitionedCall&encoder_pool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_176362"
 encoder_dropout4/PartitionedCallН
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout4/PartitionedCall:output:0encoder_conv5_17780encoder_conv5_17782*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_176602'
%encoder_conv5/StatefulPartitionedCallњ
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_174072
encoder_pool5/PartitionedCallЊ
 encoder_dropout5/PartitionedCallPartitionedCall&encoder_pool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_176942"
 encoder_dropout5/PartitionedCallЗ
flatten/PartitionedCallPartitionedCall)encoder_dropout5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_177132
flatten/PartitionedCallЮ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_17788dense_17790*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_177322
dense/StatefulPartitionedCallс
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%encoder_conv1/StatefulPartitionedCall%encoder_conv1/StatefulPartitionedCall2N
%encoder_conv2/StatefulPartitionedCall%encoder_conv2/StatefulPartitionedCall2N
%encoder_conv3/StatefulPartitionedCall%encoder_conv3/StatefulPartitionedCall2N
%encoder_conv4/StatefulPartitionedCall%encoder_conv4/StatefulPartitionedCall2N
%encoder_conv5/StatefulPartitionedCall%encoder_conv5/StatefulPartitionedCall:f b
1
_output_shapes
:         ђђ
-
_user_specified_nameencoder_conv1_input
ЛN
╠
E__inference_sequential_layer_call_and_return_conditional_losses_17749
encoder_conv1_input
encoder_conv1_17439
encoder_conv1_17441
encoder_conv2_17497
encoder_conv2_17499
encoder_conv3_17555
encoder_conv3_17557
encoder_conv4_17613
encoder_conv4_17615
encoder_conv5_17671
encoder_conv5_17673
dense_17743
dense_17745
identityѕбdense/StatefulPartitionedCallб%encoder_conv1/StatefulPartitionedCallб%encoder_conv2/StatefulPartitionedCallб%encoder_conv3/StatefulPartitionedCallб%encoder_conv4/StatefulPartitionedCallб%encoder_conv5/StatefulPartitionedCallб(encoder_dropout1/StatefulPartitionedCallб(encoder_dropout2/StatefulPartitionedCallб(encoder_dropout3/StatefulPartitionedCallб(encoder_dropout4/StatefulPartitionedCallб(encoder_dropout5/StatefulPartitionedCall┴
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCallencoder_conv1_inputencoder_conv1_17439encoder_conv1_17441*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_174282'
%encoder_conv1/StatefulPartitionedCallњ
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_173592
encoder_pool1/PartitionedCallФ
(encoder_dropout1/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_174572*
(encoder_dropout1/StatefulPartitionedCallП
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout1/StatefulPartitionedCall:output:0encoder_conv2_17497encoder_conv2_17499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_174862'
%encoder_conv2/StatefulPartitionedCallњ
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_173712
encoder_pool2/PartitionedCallо
(encoder_dropout2/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool2/PartitionedCall:output:0)^encoder_dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:            * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_175152*
(encoder_dropout2/StatefulPartitionedCallП
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout2/StatefulPartitionedCall:output:0encoder_conv3_17555encoder_conv3_17557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_175442'
%encoder_conv3/StatefulPartitionedCallњ
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_173832
encoder_pool3/PartitionedCallо
(encoder_dropout3/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool3/PartitionedCall:output:0)^encoder_dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_175732*
(encoder_dropout3/StatefulPartitionedCallП
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout3/StatefulPartitionedCall:output:0encoder_conv4_17613encoder_conv4_17615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_176022'
%encoder_conv4/StatefulPartitionedCallњ
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_173952
encoder_pool4/PartitionedCallо
(encoder_dropout4/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool4/PartitionedCall:output:0)^encoder_dropout3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_176312*
(encoder_dropout4/StatefulPartitionedCallП
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout4/StatefulPartitionedCall:output:0encoder_conv5_17671encoder_conv5_17673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_176602'
%encoder_conv5/StatefulPartitionedCallњ
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_174072
encoder_pool5/PartitionedCallо
(encoder_dropout5/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool5/PartitionedCall:output:0)^encoder_dropout4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_176892*
(encoder_dropout5/StatefulPartitionedCallЧ
flatten/PartitionedCallPartitionedCall1encoder_dropout5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_177132
flatten/PartitionedCallЮ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_17743dense_17745*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_177322
dense/StatefulPartitionedCall║
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall)^encoder_dropout1/StatefulPartitionedCall)^encoder_dropout2/StatefulPartitionedCall)^encoder_dropout3/StatefulPartitionedCall)^encoder_dropout4/StatefulPartitionedCall)^encoder_dropout5/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%encoder_conv1/StatefulPartitionedCall%encoder_conv1/StatefulPartitionedCall2N
%encoder_conv2/StatefulPartitionedCall%encoder_conv2/StatefulPartitionedCall2N
%encoder_conv3/StatefulPartitionedCall%encoder_conv3/StatefulPartitionedCall2N
%encoder_conv4/StatefulPartitionedCall%encoder_conv4/StatefulPartitionedCall2N
%encoder_conv5/StatefulPartitionedCall%encoder_conv5/StatefulPartitionedCall2T
(encoder_dropout1/StatefulPartitionedCall(encoder_dropout1/StatefulPartitionedCall2T
(encoder_dropout2/StatefulPartitionedCall(encoder_dropout2/StatefulPartitionedCall2T
(encoder_dropout3/StatefulPartitionedCall(encoder_dropout3/StatefulPartitionedCall2T
(encoder_dropout4/StatefulPartitionedCall(encoder_dropout4/StatefulPartitionedCall2T
(encoder_dropout5/StatefulPartitionedCall(encoder_dropout5/StatefulPartitionedCall:f b
1
_output_shapes
:         ђђ
-
_user_specified_nameencoder_conv1_input
Г
o
C__inference_subtract_layer_call_and_return_conditional_losses_18936
inputs_0
inputs_1
identityX
subSubinputs_0inputs_1*
T0*(
_output_shapes
:         ђ2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ђ:         ђ:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         ђ
"
_user_specified_name
inputs/1
§
d
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_17371

inputs
identityг
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ї
ѓ
-__inference_encoder_conv1_layer_call_fn_18992

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_174282
StatefulPartitionedCallў
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ђђ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ћ	
░
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_17428

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
Eluo
IdentityIdentityElu:activations:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ђђ:::Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ме
щ
!__inference__traced_restore_19506
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias
assignvariableop_2_decay$
 assignvariableop_3_learning_rate
assignvariableop_4_momentum
assignvariableop_5_rho#
assignvariableop_6_rmsprop_iter6
2assignvariableop_7_sequential_encoder_conv1_kernel4
0assignvariableop_8_sequential_encoder_conv1_bias6
2assignvariableop_9_sequential_encoder_conv2_kernel5
1assignvariableop_10_sequential_encoder_conv2_bias7
3assignvariableop_11_sequential_encoder_conv3_kernel5
1assignvariableop_12_sequential_encoder_conv3_bias7
3assignvariableop_13_sequential_encoder_conv4_kernel5
1assignvariableop_14_sequential_encoder_conv4_bias7
3assignvariableop_15_sequential_encoder_conv5_kernel5
1assignvariableop_16_sequential_encoder_conv5_bias/
+assignvariableop_17_sequential_dense_kernel-
)assignvariableop_18_sequential_dense_bias
assignvariableop_19_total
assignvariableop_20_count&
"assignvariableop_21_true_positives&
"assignvariableop_22_true_negatives'
#assignvariableop_23_false_positives'
#assignvariableop_24_false_negatives2
.assignvariableop_25_rmsprop_dense_1_kernel_rms0
,assignvariableop_26_rmsprop_dense_1_bias_rmsC
?assignvariableop_27_rmsprop_sequential_encoder_conv1_kernel_rmsA
=assignvariableop_28_rmsprop_sequential_encoder_conv1_bias_rmsC
?assignvariableop_29_rmsprop_sequential_encoder_conv2_kernel_rmsA
=assignvariableop_30_rmsprop_sequential_encoder_conv2_bias_rmsC
?assignvariableop_31_rmsprop_sequential_encoder_conv3_kernel_rmsA
=assignvariableop_32_rmsprop_sequential_encoder_conv3_bias_rmsC
?assignvariableop_33_rmsprop_sequential_encoder_conv4_kernel_rmsA
=assignvariableop_34_rmsprop_sequential_encoder_conv4_bias_rmsC
?assignvariableop_35_rmsprop_sequential_encoder_conv5_kernel_rmsA
=assignvariableop_36_rmsprop_sequential_encoder_conv5_bias_rms;
7assignvariableop_37_rmsprop_sequential_dense_kernel_rms9
5assignvariableop_38_rmsprop_sequential_dense_bias_rms
identity_40ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Е
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*х
valueФBе(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesя
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesШ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityъ
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ц
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ю
AssignVariableOp_2AssignVariableOpassignvariableop_2_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4а
AssignVariableOp_4AssignVariableOpassignvariableop_4_momentumIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Џ
AssignVariableOp_5AssignVariableOpassignvariableop_5_rhoIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6ц
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7и
AssignVariableOp_7AssignVariableOp2assignvariableop_7_sequential_encoder_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8х
AssignVariableOp_8AssignVariableOp0assignvariableop_8_sequential_encoder_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9и
AssignVariableOp_9AssignVariableOp2assignvariableop_9_sequential_encoder_conv2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╣
AssignVariableOp_10AssignVariableOp1assignvariableop_10_sequential_encoder_conv2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╗
AssignVariableOp_11AssignVariableOp3assignvariableop_11_sequential_encoder_conv3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╣
AssignVariableOp_12AssignVariableOp1assignvariableop_12_sequential_encoder_conv3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╗
AssignVariableOp_13AssignVariableOp3assignvariableop_13_sequential_encoder_conv4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╣
AssignVariableOp_14AssignVariableOp1assignvariableop_14_sequential_encoder_conv4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╗
AssignVariableOp_15AssignVariableOp3assignvariableop_15_sequential_encoder_conv5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╣
AssignVariableOp_16AssignVariableOp1assignvariableop_16_sequential_encoder_conv5_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17│
AssignVariableOp_17AssignVariableOp+assignvariableop_17_sequential_dense_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18▒
AssignVariableOp_18AssignVariableOp)assignvariableop_18_sequential_dense_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19А
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ф
AssignVariableOp_21AssignVariableOp"assignvariableop_21_true_positivesIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ф
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_negativesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ф
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ф
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Х
AssignVariableOp_25AssignVariableOp.assignvariableop_25_rmsprop_dense_1_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┤
AssignVariableOp_26AssignVariableOp,assignvariableop_26_rmsprop_dense_1_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27К
AssignVariableOp_27AssignVariableOp?assignvariableop_27_rmsprop_sequential_encoder_conv1_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28┼
AssignVariableOp_28AssignVariableOp=assignvariableop_28_rmsprop_sequential_encoder_conv1_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29К
AssignVariableOp_29AssignVariableOp?assignvariableop_29_rmsprop_sequential_encoder_conv2_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30┼
AssignVariableOp_30AssignVariableOp=assignvariableop_30_rmsprop_sequential_encoder_conv2_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31К
AssignVariableOp_31AssignVariableOp?assignvariableop_31_rmsprop_sequential_encoder_conv3_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32┼
AssignVariableOp_32AssignVariableOp=assignvariableop_32_rmsprop_sequential_encoder_conv3_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33К
AssignVariableOp_33AssignVariableOp?assignvariableop_33_rmsprop_sequential_encoder_conv4_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34┼
AssignVariableOp_34AssignVariableOp=assignvariableop_34_rmsprop_sequential_encoder_conv4_bias_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35К
AssignVariableOp_35AssignVariableOp?assignvariableop_35_rmsprop_sequential_encoder_conv5_kernel_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36┼
AssignVariableOp_36AssignVariableOp=assignvariableop_36_rmsprop_sequential_encoder_conv5_bias_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37┐
AssignVariableOp_37AssignVariableOp7assignvariableop_37_rmsprop_sequential_dense_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38й
AssignVariableOp_38AssignVariableOp5assignvariableop_38_rmsprop_sequential_dense_bias_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39Ф
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*│
_input_shapesА
ъ: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Д
I
-__inference_encoder_pool2_layer_call_fn_17377

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_173712
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_19098

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         02
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         02
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         02

Identity"
identityIdentity:output:0*.
_input_shapes
:         0:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
Д
I
-__inference_encoder_pool5_layer_call_fn_17413

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_174072
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Д
I
-__inference_encoder_pool4_layer_call_fn_17401

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_173952
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_17457

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Ќ
╚
,__inference_functional_1_layer_call_fn_18721
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_182882
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1
Ь
i
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17578

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         02

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         0:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
Д
I
-__inference_encoder_pool1_layer_call_fn_17365

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_173592
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ё
ѓ
-__inference_encoder_conv4_layer_call_fn_19133

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_176022
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_17515

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:            2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:            *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:            2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:            2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:            2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:            2

Identity"
identityIdentity:output:0*.
_input_shapes
:            :W S
/
_output_shapes
:            
 
_user_specified_nameinputs
Ѕ	
░
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_19124

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         0:::W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_19213

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
╠
i
0__inference_encoder_dropout5_layer_call_fn_19202

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_176892
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
Ё
ѓ
-__inference_encoder_conv3_layer_call_fn_19086

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_175442
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:           02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:            ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:            
 
_user_specified_nameinputs
╔
j
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17689

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         P2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         P2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         P2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
Ь
i
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_19150

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Љ
к
,__inference_functional_1_layer_call_fn_18319
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_182882
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:         ђђ
!
_user_specified_name	input_2
Ѕ	
░
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_19030

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@:::W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_17713

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:         P:W S
/
_output_shapes
:         P
 
_user_specified_nameinputs
§
d
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_17359

inputs
identityг
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
з
љ
G__inference_functional_1_layer_call_and_return_conditional_losses_18150
input_1
input_2
sequential_18104
sequential_18106
sequential_18108
sequential_18110
sequential_18112
sequential_18114
sequential_18116
sequential_18118
sequential_18120
sequential_18122
sequential_18124
sequential_18126
dense_1_18144
dense_1_18146
identityѕбdense_1/StatefulPartitionedCallб"sequential/StatefulPartitionedCallб$sequential/StatefulPartitionedCall_1т
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_18104sequential_18106sequential_18108sequential_18110sequential_18112sequential_18114sequential_18116sequential_18118sequential_18120sequential_18122sequential_18124sequential_18126*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_179162$
"sequential/StatefulPartitionedCallж
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_18104sequential_18106sequential_18108sequential_18110sequential_18112sequential_18114sequential_18116sequential_18118sequential_18120sequential_18122sequential_18124sequential_18126*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_179162&
$sequential/StatefulPartitionedCall_1Е
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_180502
subtract/PartitionedCallё
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_180642!
tf_op_layer_Abs/PartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_18144dense_1_18146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_180832!
dense_1/StatefulPartitionedCallЖ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:         ђђ
!
_user_specified_name	input_2
Ь
i
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_19056

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:            2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:            2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:            :W S
/
_output_shapes
:            
 
_user_specified_nameinputs
и	
ќ
*__inference_sequential_layer_call_fn_18901

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178422
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
╠
i
0__inference_encoder_dropout1_layer_call_fn_19014

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_174572
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
чб
А
G__inference_functional_1_layer_call_and_return_conditional_losses_18543
inputs_0
inputs_1;
7sequential_encoder_conv1_conv2d_readvariableop_resource<
8sequential_encoder_conv1_biasadd_readvariableop_resource;
7sequential_encoder_conv2_conv2d_readvariableop_resource<
8sequential_encoder_conv2_biasadd_readvariableop_resource;
7sequential_encoder_conv3_conv2d_readvariableop_resource<
8sequential_encoder_conv3_biasadd_readvariableop_resource;
7sequential_encoder_conv4_conv2d_readvariableop_resource<
8sequential_encoder_conv4_biasadd_readvariableop_resource;
7sequential_encoder_conv5_conv2d_readvariableop_resource<
8sequential_encoder_conv5_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕЯ
.sequential/encoder_conv1/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.sequential/encoder_conv1/Conv2D/ReadVariableOpЫ
sequential/encoder_conv1/Conv2DConv2Dinputs_06sequential/encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2!
sequential/encoder_conv1/Conv2DО
/sequential/encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/encoder_conv1/BiasAdd/ReadVariableOpЬ
 sequential/encoder_conv1/BiasAddBiasAdd(sequential/encoder_conv1/Conv2D:output:07sequential/encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2"
 sequential/encoder_conv1/BiasAddф
sequential/encoder_conv1/EluElu)sequential/encoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
sequential/encoder_conv1/Eluу
 sequential/encoder_pool1/MaxPoolMaxPool*sequential/encoder_conv1/Elu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool1/MaxPoolЏ
)sequential/encoder_dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)sequential/encoder_dropout1/dropout/ConstЫ
'sequential/encoder_dropout1/dropout/MulMul)sequential/encoder_pool1/MaxPool:output:02sequential/encoder_dropout1/dropout/Const:output:0*
T0*/
_output_shapes
:         @@2)
'sequential/encoder_dropout1/dropout/Mul»
)sequential/encoder_dropout1/dropout/ShapeShape)sequential/encoder_pool1/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout1/dropout/Shapeљ
@sequential/encoder_dropout1/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02B
@sequential/encoder_dropout1/dropout/random_uniform/RandomUniformГ
2sequential/encoder_dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2sequential/encoder_dropout1/dropout/GreaterEqual/yХ
0sequential/encoder_dropout1/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout1/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @@22
0sequential/encoder_dropout1/dropout/GreaterEqual█
(sequential/encoder_dropout1/dropout/CastCast4sequential/encoder_dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2*
(sequential/encoder_dropout1/dropout/CastЫ
)sequential/encoder_dropout1/dropout/Mul_1Mul+sequential/encoder_dropout1/dropout/Mul:z:0,sequential/encoder_dropout1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2+
)sequential/encoder_dropout1/dropout/Mul_1Я
.sequential/encoder_conv2/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential/encoder_conv2/Conv2D/ReadVariableOpЋ
sequential/encoder_conv2/Conv2DConv2D-sequential/encoder_dropout1/dropout/Mul_1:z:06sequential/encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2!
sequential/encoder_conv2/Conv2DО
/sequential/encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/encoder_conv2/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv2/BiasAddBiasAdd(sequential/encoder_conv2/Conv2D:output:07sequential/encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2"
 sequential/encoder_conv2/BiasAddе
sequential/encoder_conv2/EluElu)sequential/encoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
sequential/encoder_conv2/Eluу
 sequential/encoder_pool2/MaxPoolMaxPool*sequential/encoder_conv2/Elu:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool2/MaxPoolЏ
)sequential/encoder_dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)sequential/encoder_dropout2/dropout/ConstЫ
'sequential/encoder_dropout2/dropout/MulMul)sequential/encoder_pool2/MaxPool:output:02sequential/encoder_dropout2/dropout/Const:output:0*
T0*/
_output_shapes
:            2)
'sequential/encoder_dropout2/dropout/Mul»
)sequential/encoder_dropout2/dropout/ShapeShape)sequential/encoder_pool2/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout2/dropout/Shapeљ
@sequential/encoder_dropout2/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout2/dropout/Shape:output:0*
T0*/
_output_shapes
:            *
dtype02B
@sequential/encoder_dropout2/dropout/random_uniform/RandomUniformГ
2sequential/encoder_dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2sequential/encoder_dropout2/dropout/GreaterEqual/yХ
0sequential/encoder_dropout2/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout2/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:            22
0sequential/encoder_dropout2/dropout/GreaterEqual█
(sequential/encoder_dropout2/dropout/CastCast4sequential/encoder_dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:            2*
(sequential/encoder_dropout2/dropout/CastЫ
)sequential/encoder_dropout2/dropout/Mul_1Mul+sequential/encoder_dropout2/dropout/Mul:z:0,sequential/encoder_dropout2/dropout/Cast:y:0*
T0*/
_output_shapes
:            2+
)sequential/encoder_dropout2/dropout/Mul_1Я
.sequential/encoder_conv3/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype020
.sequential/encoder_conv3/Conv2D/ReadVariableOpЋ
sequential/encoder_conv3/Conv2DConv2D-sequential/encoder_dropout2/dropout/Mul_1:z:06sequential/encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2!
sequential/encoder_conv3/Conv2DО
/sequential/encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/encoder_conv3/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv3/BiasAddBiasAdd(sequential/encoder_conv3/Conv2D:output:07sequential/encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02"
 sequential/encoder_conv3/BiasAddе
sequential/encoder_conv3/EluElu)sequential/encoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:           02
sequential/encoder_conv3/Eluу
 sequential/encoder_pool3/MaxPoolMaxPool*sequential/encoder_conv3/Elu:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool3/MaxPoolЏ
)sequential/encoder_dropout3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)sequential/encoder_dropout3/dropout/ConstЫ
'sequential/encoder_dropout3/dropout/MulMul)sequential/encoder_pool3/MaxPool:output:02sequential/encoder_dropout3/dropout/Const:output:0*
T0*/
_output_shapes
:         02)
'sequential/encoder_dropout3/dropout/Mul»
)sequential/encoder_dropout3/dropout/ShapeShape)sequential/encoder_pool3/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout3/dropout/Shapeљ
@sequential/encoder_dropout3/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout3/dropout/Shape:output:0*
T0*/
_output_shapes
:         0*
dtype02B
@sequential/encoder_dropout3/dropout/random_uniform/RandomUniformГ
2sequential/encoder_dropout3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2sequential/encoder_dropout3/dropout/GreaterEqual/yХ
0sequential/encoder_dropout3/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout3/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         022
0sequential/encoder_dropout3/dropout/GreaterEqual█
(sequential/encoder_dropout3/dropout/CastCast4sequential/encoder_dropout3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         02*
(sequential/encoder_dropout3/dropout/CastЫ
)sequential/encoder_dropout3/dropout/Mul_1Mul+sequential/encoder_dropout3/dropout/Mul:z:0,sequential/encoder_dropout3/dropout/Cast:y:0*
T0*/
_output_shapes
:         02+
)sequential/encoder_dropout3/dropout/Mul_1Я
.sequential/encoder_conv4/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype020
.sequential/encoder_conv4/Conv2D/ReadVariableOpЋ
sequential/encoder_conv4/Conv2DConv2D-sequential/encoder_dropout3/dropout/Mul_1:z:06sequential/encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2!
sequential/encoder_conv4/Conv2DО
/sequential/encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/encoder_conv4/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv4/BiasAddBiasAdd(sequential/encoder_conv4/Conv2D:output:07sequential/encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2"
 sequential/encoder_conv4/BiasAddе
sequential/encoder_conv4/EluElu)sequential/encoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
sequential/encoder_conv4/Eluу
 sequential/encoder_pool4/MaxPoolMaxPool*sequential/encoder_conv4/Elu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool4/MaxPoolЏ
)sequential/encoder_dropout4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)sequential/encoder_dropout4/dropout/ConstЫ
'sequential/encoder_dropout4/dropout/MulMul)sequential/encoder_pool4/MaxPool:output:02sequential/encoder_dropout4/dropout/Const:output:0*
T0*/
_output_shapes
:         @2)
'sequential/encoder_dropout4/dropout/Mul»
)sequential/encoder_dropout4/dropout/ShapeShape)sequential/encoder_pool4/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout4/dropout/Shapeљ
@sequential/encoder_dropout4/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout4/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02B
@sequential/encoder_dropout4/dropout/random_uniform/RandomUniformГ
2sequential/encoder_dropout4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2sequential/encoder_dropout4/dropout/GreaterEqual/yХ
0sequential/encoder_dropout4/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout4/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @22
0sequential/encoder_dropout4/dropout/GreaterEqual█
(sequential/encoder_dropout4/dropout/CastCast4sequential/encoder_dropout4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2*
(sequential/encoder_dropout4/dropout/CastЫ
)sequential/encoder_dropout4/dropout/Mul_1Mul+sequential/encoder_dropout4/dropout/Mul:z:0,sequential/encoder_dropout4/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2+
)sequential/encoder_dropout4/dropout/Mul_1Я
.sequential/encoder_conv5/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype020
.sequential/encoder_conv5/Conv2D/ReadVariableOpЋ
sequential/encoder_conv5/Conv2DConv2D-sequential/encoder_dropout4/dropout/Mul_1:z:06sequential/encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2!
sequential/encoder_conv5/Conv2DО
/sequential/encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype021
/sequential/encoder_conv5/BiasAdd/ReadVariableOpВ
 sequential/encoder_conv5/BiasAddBiasAdd(sequential/encoder_conv5/Conv2D:output:07sequential/encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2"
 sequential/encoder_conv5/BiasAddе
sequential/encoder_conv5/EluElu)sequential/encoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:         P2
sequential/encoder_conv5/Eluу
 sequential/encoder_pool5/MaxPoolMaxPool*sequential/encoder_conv5/Elu:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool5/MaxPoolЏ
)sequential/encoder_dropout5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)sequential/encoder_dropout5/dropout/ConstЫ
'sequential/encoder_dropout5/dropout/MulMul)sequential/encoder_pool5/MaxPool:output:02sequential/encoder_dropout5/dropout/Const:output:0*
T0*/
_output_shapes
:         P2)
'sequential/encoder_dropout5/dropout/Mul»
)sequential/encoder_dropout5/dropout/ShapeShape)sequential/encoder_pool5/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout5/dropout/Shapeљ
@sequential/encoder_dropout5/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout5/dropout/Shape:output:0*
T0*/
_output_shapes
:         P*
dtype02B
@sequential/encoder_dropout5/dropout/random_uniform/RandomUniformГ
2sequential/encoder_dropout5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2sequential/encoder_dropout5/dropout/GreaterEqual/yХ
0sequential/encoder_dropout5/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout5/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         P22
0sequential/encoder_dropout5/dropout/GreaterEqual█
(sequential/encoder_dropout5/dropout/CastCast4sequential/encoder_dropout5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         P2*
(sequential/encoder_dropout5/dropout/CastЫ
)sequential/encoder_dropout5/dropout/Mul_1Mul+sequential/encoder_dropout5/dropout/Mul:z:0,sequential/encoder_dropout5/dropout/Cast:y:0*
T0*/
_output_shapes
:         P2+
)sequential/encoder_dropout5/dropout/Mul_1Ё
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
sequential/flatten/Const╚
sequential/flatten/ReshapeReshape-sequential/encoder_dropout5/dropout/Mul_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ
2
sequential/flatten/Reshape┬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02(
&sequential/dense/MatMul/ReadVariableOp─
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/MatMul└
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpк
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/BiasAddЋ
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/SigmoidС
0sequential/encoder_conv1/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0sequential/encoder_conv1/Conv2D_1/ReadVariableOpЭ
!sequential/encoder_conv1/Conv2D_1Conv2Dinputs_18sequential/encoder_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2#
!sequential/encoder_conv1/Conv2D_1█
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOpШ
"sequential/encoder_conv1/BiasAdd_1BiasAdd*sequential/encoder_conv1/Conv2D_1:output:09sequential/encoder_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2$
"sequential/encoder_conv1/BiasAdd_1░
sequential/encoder_conv1/Elu_1Elu+sequential/encoder_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:         ђђ2 
sequential/encoder_conv1/Elu_1ь
"sequential/encoder_pool1/MaxPool_1MaxPool,sequential/encoder_conv1/Elu_1:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool1/MaxPool_1Ъ
+sequential/encoder_dropout1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+sequential/encoder_dropout1/dropout_1/ConstЩ
)sequential/encoder_dropout1/dropout_1/MulMul+sequential/encoder_pool1/MaxPool_1:output:04sequential/encoder_dropout1/dropout_1/Const:output:0*
T0*/
_output_shapes
:         @@2+
)sequential/encoder_dropout1/dropout_1/Mulх
+sequential/encoder_dropout1/dropout_1/ShapeShape+sequential/encoder_pool1/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout1/dropout_1/Shapeќ
Bsequential/encoder_dropout1/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout1/dropout_1/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype02D
Bsequential/encoder_dropout1/dropout_1/random_uniform/RandomUniform▒
4sequential/encoder_dropout1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4sequential/encoder_dropout1/dropout_1/GreaterEqual/yЙ
2sequential/encoder_dropout1/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout1/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout1/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @@24
2sequential/encoder_dropout1/dropout_1/GreaterEqualр
*sequential/encoder_dropout1/dropout_1/CastCast6sequential/encoder_dropout1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2,
*sequential/encoder_dropout1/dropout_1/CastЩ
+sequential/encoder_dropout1/dropout_1/Mul_1Mul-sequential/encoder_dropout1/dropout_1/Mul:z:0.sequential/encoder_dropout1/dropout_1/Cast:y:0*
T0*/
_output_shapes
:         @@2-
+sequential/encoder_dropout1/dropout_1/Mul_1С
0sequential/encoder_conv2/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0sequential/encoder_conv2/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv2/Conv2D_1Conv2D/sequential/encoder_dropout1/dropout_1/Mul_1:z:08sequential/encoder_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2#
!sequential/encoder_conv2/Conv2D_1█
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv2/BiasAdd_1BiasAdd*sequential/encoder_conv2/Conv2D_1:output:09sequential/encoder_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2$
"sequential/encoder_conv2/BiasAdd_1«
sequential/encoder_conv2/Elu_1Elu+sequential/encoder_conv2/BiasAdd_1:output:0*
T0*/
_output_shapes
:         @@ 2 
sequential/encoder_conv2/Elu_1ь
"sequential/encoder_pool2/MaxPool_1MaxPool,sequential/encoder_conv2/Elu_1:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool2/MaxPool_1Ъ
+sequential/encoder_dropout2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+sequential/encoder_dropout2/dropout_1/ConstЩ
)sequential/encoder_dropout2/dropout_1/MulMul+sequential/encoder_pool2/MaxPool_1:output:04sequential/encoder_dropout2/dropout_1/Const:output:0*
T0*/
_output_shapes
:            2+
)sequential/encoder_dropout2/dropout_1/Mulх
+sequential/encoder_dropout2/dropout_1/ShapeShape+sequential/encoder_pool2/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout2/dropout_1/Shapeќ
Bsequential/encoder_dropout2/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout2/dropout_1/Shape:output:0*
T0*/
_output_shapes
:            *
dtype02D
Bsequential/encoder_dropout2/dropout_1/random_uniform/RandomUniform▒
4sequential/encoder_dropout2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4sequential/encoder_dropout2/dropout_1/GreaterEqual/yЙ
2sequential/encoder_dropout2/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout2/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout2/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:            24
2sequential/encoder_dropout2/dropout_1/GreaterEqualр
*sequential/encoder_dropout2/dropout_1/CastCast6sequential/encoder_dropout2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:            2,
*sequential/encoder_dropout2/dropout_1/CastЩ
+sequential/encoder_dropout2/dropout_1/Mul_1Mul-sequential/encoder_dropout2/dropout_1/Mul:z:0.sequential/encoder_dropout2/dropout_1/Cast:y:0*
T0*/
_output_shapes
:            2-
+sequential/encoder_dropout2/dropout_1/Mul_1С
0sequential/encoder_conv3/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype022
0sequential/encoder_conv3/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv3/Conv2D_1Conv2D/sequential/encoder_dropout2/dropout_1/Mul_1:z:08sequential/encoder_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2#
!sequential/encoder_conv3/Conv2D_1█
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv3/BiasAdd_1BiasAdd*sequential/encoder_conv3/Conv2D_1:output:09sequential/encoder_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02$
"sequential/encoder_conv3/BiasAdd_1«
sequential/encoder_conv3/Elu_1Elu+sequential/encoder_conv3/BiasAdd_1:output:0*
T0*/
_output_shapes
:           02 
sequential/encoder_conv3/Elu_1ь
"sequential/encoder_pool3/MaxPool_1MaxPool,sequential/encoder_conv3/Elu_1:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool3/MaxPool_1Ъ
+sequential/encoder_dropout3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+sequential/encoder_dropout3/dropout_1/ConstЩ
)sequential/encoder_dropout3/dropout_1/MulMul+sequential/encoder_pool3/MaxPool_1:output:04sequential/encoder_dropout3/dropout_1/Const:output:0*
T0*/
_output_shapes
:         02+
)sequential/encoder_dropout3/dropout_1/Mulх
+sequential/encoder_dropout3/dropout_1/ShapeShape+sequential/encoder_pool3/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout3/dropout_1/Shapeќ
Bsequential/encoder_dropout3/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout3/dropout_1/Shape:output:0*
T0*/
_output_shapes
:         0*
dtype02D
Bsequential/encoder_dropout3/dropout_1/random_uniform/RandomUniform▒
4sequential/encoder_dropout3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4sequential/encoder_dropout3/dropout_1/GreaterEqual/yЙ
2sequential/encoder_dropout3/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout3/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout3/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         024
2sequential/encoder_dropout3/dropout_1/GreaterEqualр
*sequential/encoder_dropout3/dropout_1/CastCast6sequential/encoder_dropout3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         02,
*sequential/encoder_dropout3/dropout_1/CastЩ
+sequential/encoder_dropout3/dropout_1/Mul_1Mul-sequential/encoder_dropout3/dropout_1/Mul:z:0.sequential/encoder_dropout3/dropout_1/Cast:y:0*
T0*/
_output_shapes
:         02-
+sequential/encoder_dropout3/dropout_1/Mul_1С
0sequential/encoder_conv4/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype022
0sequential/encoder_conv4/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv4/Conv2D_1Conv2D/sequential/encoder_dropout3/dropout_1/Mul_1:z:08sequential/encoder_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2#
!sequential/encoder_conv4/Conv2D_1█
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv4/BiasAdd_1BiasAdd*sequential/encoder_conv4/Conv2D_1:output:09sequential/encoder_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2$
"sequential/encoder_conv4/BiasAdd_1«
sequential/encoder_conv4/Elu_1Elu+sequential/encoder_conv4/BiasAdd_1:output:0*
T0*/
_output_shapes
:         @2 
sequential/encoder_conv4/Elu_1ь
"sequential/encoder_pool4/MaxPool_1MaxPool,sequential/encoder_conv4/Elu_1:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool4/MaxPool_1Ъ
+sequential/encoder_dropout4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+sequential/encoder_dropout4/dropout_1/ConstЩ
)sequential/encoder_dropout4/dropout_1/MulMul+sequential/encoder_pool4/MaxPool_1:output:04sequential/encoder_dropout4/dropout_1/Const:output:0*
T0*/
_output_shapes
:         @2+
)sequential/encoder_dropout4/dropout_1/Mulх
+sequential/encoder_dropout4/dropout_1/ShapeShape+sequential/encoder_pool4/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout4/dropout_1/Shapeќ
Bsequential/encoder_dropout4/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout4/dropout_1/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02D
Bsequential/encoder_dropout4/dropout_1/random_uniform/RandomUniform▒
4sequential/encoder_dropout4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4sequential/encoder_dropout4/dropout_1/GreaterEqual/yЙ
2sequential/encoder_dropout4/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout4/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout4/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @24
2sequential/encoder_dropout4/dropout_1/GreaterEqualр
*sequential/encoder_dropout4/dropout_1/CastCast6sequential/encoder_dropout4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2,
*sequential/encoder_dropout4/dropout_1/CastЩ
+sequential/encoder_dropout4/dropout_1/Mul_1Mul-sequential/encoder_dropout4/dropout_1/Mul:z:0.sequential/encoder_dropout4/dropout_1/Cast:y:0*
T0*/
_output_shapes
:         @2-
+sequential/encoder_dropout4/dropout_1/Mul_1С
0sequential/encoder_conv5/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype022
0sequential/encoder_conv5/Conv2D_1/ReadVariableOpЮ
!sequential/encoder_conv5/Conv2D_1Conv2D/sequential/encoder_dropout4/dropout_1/Mul_1:z:08sequential/encoder_conv5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2#
!sequential/encoder_conv5/Conv2D_1█
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype023
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOpЗ
"sequential/encoder_conv5/BiasAdd_1BiasAdd*sequential/encoder_conv5/Conv2D_1:output:09sequential/encoder_conv5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2$
"sequential/encoder_conv5/BiasAdd_1«
sequential/encoder_conv5/Elu_1Elu+sequential/encoder_conv5/BiasAdd_1:output:0*
T0*/
_output_shapes
:         P2 
sequential/encoder_conv5/Elu_1ь
"sequential/encoder_pool5/MaxPool_1MaxPool,sequential/encoder_conv5/Elu_1:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool5/MaxPool_1Ъ
+sequential/encoder_dropout5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+sequential/encoder_dropout5/dropout_1/ConstЩ
)sequential/encoder_dropout5/dropout_1/MulMul+sequential/encoder_pool5/MaxPool_1:output:04sequential/encoder_dropout5/dropout_1/Const:output:0*
T0*/
_output_shapes
:         P2+
)sequential/encoder_dropout5/dropout_1/Mulх
+sequential/encoder_dropout5/dropout_1/ShapeShape+sequential/encoder_pool5/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout5/dropout_1/Shapeќ
Bsequential/encoder_dropout5/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout5/dropout_1/Shape:output:0*
T0*/
_output_shapes
:         P*
dtype02D
Bsequential/encoder_dropout5/dropout_1/random_uniform/RandomUniform▒
4sequential/encoder_dropout5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4sequential/encoder_dropout5/dropout_1/GreaterEqual/yЙ
2sequential/encoder_dropout5/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout5/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout5/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         P24
2sequential/encoder_dropout5/dropout_1/GreaterEqualр
*sequential/encoder_dropout5/dropout_1/CastCast6sequential/encoder_dropout5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         P2,
*sequential/encoder_dropout5/dropout_1/CastЩ
+sequential/encoder_dropout5/dropout_1/Mul_1Mul-sequential/encoder_dropout5/dropout_1/Mul:z:0.sequential/encoder_dropout5/dropout_1/Cast:y:0*
T0*/
_output_shapes
:         P2-
+sequential/encoder_dropout5/dropout_1/Mul_1Ѕ
sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
sequential/flatten/Const_1л
sequential/flatten/Reshape_1Reshape/sequential/encoder_dropout5/dropout_1/Mul_1:z:0#sequential/flatten/Const_1:output:0*
T0*(
_output_shapes
:         ђ
2
sequential/flatten/Reshape_1к
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02*
(sequential/dense/MatMul_1/ReadVariableOp╠
sequential/dense/MatMul_1MatMul%sequential/flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/MatMul_1─
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)sequential/dense/BiasAdd_1/ReadVariableOp╬
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/BiasAdd_1Џ
sequential/dense/Sigmoid_1Sigmoid#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:         ђ2
sequential/dense/Sigmoid_1ћ
subtract/subSubsequential/dense/Sigmoid:y:0sequential/dense/Sigmoid_1:y:0*
T0*(
_output_shapes
:         ђ2
subtract/subЁ
tf_op_layer_Abs/AbsAbssubtract/sub:z:0*
T0*
_cloned(*(
_output_shapes
:         ђ2
tf_op_layer_Abs/Absд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
dense_1/MatMul/ReadVariableOpю
dense_1/MatMulMatMultf_op_layer_Abs/Abs:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/Sigmoidg
IdentityIdentitydense_1/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ё
_input_shapest
r:         ђђ:         ђђ:::::::::::::::[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1
я	
Б
*__inference_sequential_layer_call_fn_17943
encoder_conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallencoder_conv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_179162
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:         ђђ
-
_user_specified_nameencoder_conv1_input
Ѕ	
░
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_17602

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         0:::W S
/
_output_shapes
:         0
 
_user_specified_nameinputs
Ѕ	
░
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_17544

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:           02
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:           02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:            :::W S
/
_output_shapes
:            
 
_user_specified_nameinputs
┌
|
'__inference_dense_1_layer_call_fn_18972

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_180832
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Пx
▓
E__inference_sequential_layer_call_and_return_conditional_losses_18814

inputs0
,encoder_conv1_conv2d_readvariableop_resource1
-encoder_conv1_biasadd_readvariableop_resource0
,encoder_conv2_conv2d_readvariableop_resource1
-encoder_conv2_biasadd_readvariableop_resource0
,encoder_conv3_conv2d_readvariableop_resource1
-encoder_conv3_biasadd_readvariableop_resource0
,encoder_conv4_conv2d_readvariableop_resource1
-encoder_conv4_biasadd_readvariableop_resource0
,encoder_conv5_conv2d_readvariableop_resource1
-encoder_conv5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityѕ┐
#encoder_conv1/Conv2D/ReadVariableOpReadVariableOp,encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#encoder_conv1/Conv2D/ReadVariableOp¤
encoder_conv1/Conv2DConv2Dinputs+encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
2
encoder_conv1/Conv2DХ
$encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder_conv1/BiasAdd/ReadVariableOp┬
encoder_conv1/BiasAddBiasAddencoder_conv1/Conv2D:output:0,encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
encoder_conv1/BiasAddЅ
encoder_conv1/EluEluencoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ2
encoder_conv1/Eluк
encoder_pool1/MaxPoolMaxPoolencoder_conv1/Elu:activations:0*/
_output_shapes
:         @@*
ksize
*
paddingSAME*
strides
2
encoder_pool1/MaxPoolЁ
encoder_dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
encoder_dropout1/dropout/Constк
encoder_dropout1/dropout/MulMulencoder_pool1/MaxPool:output:0'encoder_dropout1/dropout/Const:output:0*
T0*/
_output_shapes
:         @@2
encoder_dropout1/dropout/Mulј
encoder_dropout1/dropout/ShapeShapeencoder_pool1/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout1/dropout/Shape№
5encoder_dropout1/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @@*
dtype027
5encoder_dropout1/dropout/random_uniform/RandomUniformЌ
'encoder_dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder_dropout1/dropout/GreaterEqual/yі
%encoder_dropout1/dropout/GreaterEqualGreaterEqual>encoder_dropout1/dropout/random_uniform/RandomUniform:output:00encoder_dropout1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @@2'
%encoder_dropout1/dropout/GreaterEqual║
encoder_dropout1/dropout/CastCast)encoder_dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@2
encoder_dropout1/dropout/Castк
encoder_dropout1/dropout/Mul_1Mul encoder_dropout1/dropout/Mul:z:0!encoder_dropout1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @@2 
encoder_dropout1/dropout/Mul_1┐
#encoder_conv2/Conv2D/ReadVariableOpReadVariableOp,encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#encoder_conv2/Conv2D/ReadVariableOpж
encoder_conv2/Conv2DConv2D"encoder_dropout1/dropout/Mul_1:z:0+encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
encoder_conv2/Conv2DХ
$encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$encoder_conv2/BiasAdd/ReadVariableOp└
encoder_conv2/BiasAddBiasAddencoder_conv2/Conv2D:output:0,encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
encoder_conv2/BiasAddЄ
encoder_conv2/EluEluencoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @@ 2
encoder_conv2/Eluк
encoder_pool2/MaxPoolMaxPoolencoder_conv2/Elu:activations:0*/
_output_shapes
:            *
ksize
*
paddingSAME*
strides
2
encoder_pool2/MaxPoolЁ
encoder_dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
encoder_dropout2/dropout/Constк
encoder_dropout2/dropout/MulMulencoder_pool2/MaxPool:output:0'encoder_dropout2/dropout/Const:output:0*
T0*/
_output_shapes
:            2
encoder_dropout2/dropout/Mulј
encoder_dropout2/dropout/ShapeShapeencoder_pool2/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout2/dropout/Shape№
5encoder_dropout2/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout2/dropout/Shape:output:0*
T0*/
_output_shapes
:            *
dtype027
5encoder_dropout2/dropout/random_uniform/RandomUniformЌ
'encoder_dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder_dropout2/dropout/GreaterEqual/yі
%encoder_dropout2/dropout/GreaterEqualGreaterEqual>encoder_dropout2/dropout/random_uniform/RandomUniform:output:00encoder_dropout2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:            2'
%encoder_dropout2/dropout/GreaterEqual║
encoder_dropout2/dropout/CastCast)encoder_dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:            2
encoder_dropout2/dropout/Castк
encoder_dropout2/dropout/Mul_1Mul encoder_dropout2/dropout/Mul:z:0!encoder_dropout2/dropout/Cast:y:0*
T0*/
_output_shapes
:            2 
encoder_dropout2/dropout/Mul_1┐
#encoder_conv3/Conv2D/ReadVariableOpReadVariableOp,encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02%
#encoder_conv3/Conv2D/ReadVariableOpж
encoder_conv3/Conv2DConv2D"encoder_dropout2/dropout/Mul_1:z:0+encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           0*
paddingSAME*
strides
2
encoder_conv3/Conv2DХ
$encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$encoder_conv3/BiasAdd/ReadVariableOp└
encoder_conv3/BiasAddBiasAddencoder_conv3/Conv2D:output:0,encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           02
encoder_conv3/BiasAddЄ
encoder_conv3/EluEluencoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:           02
encoder_conv3/Eluк
encoder_pool3/MaxPoolMaxPoolencoder_conv3/Elu:activations:0*/
_output_shapes
:         0*
ksize
*
paddingSAME*
strides
2
encoder_pool3/MaxPoolЁ
encoder_dropout3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
encoder_dropout3/dropout/Constк
encoder_dropout3/dropout/MulMulencoder_pool3/MaxPool:output:0'encoder_dropout3/dropout/Const:output:0*
T0*/
_output_shapes
:         02
encoder_dropout3/dropout/Mulј
encoder_dropout3/dropout/ShapeShapeencoder_pool3/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout3/dropout/Shape№
5encoder_dropout3/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout3/dropout/Shape:output:0*
T0*/
_output_shapes
:         0*
dtype027
5encoder_dropout3/dropout/random_uniform/RandomUniformЌ
'encoder_dropout3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder_dropout3/dropout/GreaterEqual/yі
%encoder_dropout3/dropout/GreaterEqualGreaterEqual>encoder_dropout3/dropout/random_uniform/RandomUniform:output:00encoder_dropout3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         02'
%encoder_dropout3/dropout/GreaterEqual║
encoder_dropout3/dropout/CastCast)encoder_dropout3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         02
encoder_dropout3/dropout/Castк
encoder_dropout3/dropout/Mul_1Mul encoder_dropout3/dropout/Mul:z:0!encoder_dropout3/dropout/Cast:y:0*
T0*/
_output_shapes
:         02 
encoder_dropout3/dropout/Mul_1┐
#encoder_conv4/Conv2D/ReadVariableOpReadVariableOp,encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02%
#encoder_conv4/Conv2D/ReadVariableOpж
encoder_conv4/Conv2DConv2D"encoder_dropout3/dropout/Mul_1:z:0+encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
encoder_conv4/Conv2DХ
$encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$encoder_conv4/BiasAdd/ReadVariableOp└
encoder_conv4/BiasAddBiasAddencoder_conv4/Conv2D:output:0,encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
encoder_conv4/BiasAddЄ
encoder_conv4/EluEluencoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
encoder_conv4/Eluк
encoder_pool4/MaxPoolMaxPoolencoder_conv4/Elu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
encoder_pool4/MaxPoolЁ
encoder_dropout4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
encoder_dropout4/dropout/Constк
encoder_dropout4/dropout/MulMulencoder_pool4/MaxPool:output:0'encoder_dropout4/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
encoder_dropout4/dropout/Mulј
encoder_dropout4/dropout/ShapeShapeencoder_pool4/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout4/dropout/Shape№
5encoder_dropout4/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout4/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype027
5encoder_dropout4/dropout/random_uniform/RandomUniformЌ
'encoder_dropout4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder_dropout4/dropout/GreaterEqual/yі
%encoder_dropout4/dropout/GreaterEqualGreaterEqual>encoder_dropout4/dropout/random_uniform/RandomUniform:output:00encoder_dropout4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2'
%encoder_dropout4/dropout/GreaterEqual║
encoder_dropout4/dropout/CastCast)encoder_dropout4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
encoder_dropout4/dropout/Castк
encoder_dropout4/dropout/Mul_1Mul encoder_dropout4/dropout/Mul:z:0!encoder_dropout4/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2 
encoder_dropout4/dropout/Mul_1┐
#encoder_conv5/Conv2D/ReadVariableOpReadVariableOp,encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02%
#encoder_conv5/Conv2D/ReadVariableOpж
encoder_conv5/Conv2DConv2D"encoder_dropout4/dropout/Mul_1:z:0+encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P*
paddingSAME*
strides
2
encoder_conv5/Conv2DХ
$encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder_conv5/BiasAdd/ReadVariableOp└
encoder_conv5/BiasAddBiasAddencoder_conv5/Conv2D:output:0,encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         P2
encoder_conv5/BiasAddЄ
encoder_conv5/EluEluencoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:         P2
encoder_conv5/Eluк
encoder_pool5/MaxPoolMaxPoolencoder_conv5/Elu:activations:0*/
_output_shapes
:         P*
ksize
*
paddingSAME*
strides
2
encoder_pool5/MaxPoolЁ
encoder_dropout5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
encoder_dropout5/dropout/Constк
encoder_dropout5/dropout/MulMulencoder_pool5/MaxPool:output:0'encoder_dropout5/dropout/Const:output:0*
T0*/
_output_shapes
:         P2
encoder_dropout5/dropout/Mulј
encoder_dropout5/dropout/ShapeShapeencoder_pool5/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout5/dropout/Shape№
5encoder_dropout5/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout5/dropout/Shape:output:0*
T0*/
_output_shapes
:         P*
dtype027
5encoder_dropout5/dropout/random_uniform/RandomUniformЌ
'encoder_dropout5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder_dropout5/dropout/GreaterEqual/yі
%encoder_dropout5/dropout/GreaterEqualGreaterEqual>encoder_dropout5/dropout/random_uniform/RandomUniform:output:00encoder_dropout5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         P2'
%encoder_dropout5/dropout/GreaterEqual║
encoder_dropout5/dropout/CastCast)encoder_dropout5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         P2
encoder_dropout5/dropout/Castк
encoder_dropout5/dropout/Mul_1Mul encoder_dropout5/dropout/Mul:z:0!encoder_dropout5/dropout/Cast:y:0*
T0*/
_output_shapes
:         P2 
encoder_dropout5/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/Constю
flatten/ReshapeReshape"encoder_dropout5/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ
2
flatten/ReshapeА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђ
ђ*
dtype02
dense/MatMul/ReadVariableOpў
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/MatMulЪ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
dense/BiasAdd/ReadVariableOpџ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense/Sigmoidf
IdentityIdentitydense/Sigmoid:y:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ђђ:::::::::::::Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ч
serving_defaultу
E
input_1:
serving_default_input_1:0         ђђ
E
input_2:
serving_default_input_2:0         ђђ;
dense_10
StatefulPartitionedCall:0         tensorflow/serving/predict:╬у
Ыє
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
+ъ&call_and_return_all_conditional_losses
Ъ_default_save_signature
а__call__"Аё
_tf_keras_networkёё{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_conv1_input"}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Abs", "trainable": true, "dtype": "float32", "node_def": {"name": "Abs", "op": "Abs", "input": ["subtract/sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Abs", "inbound_nodes": [[["subtract", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["tf_op_layer_Abs", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 1]}, {"class_name": "TensorShape", "items": [null, 128, 128, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_conv1_input"}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Abs", "trainable": true, "dtype": "float32", "node_def": {"name": "Abs", "op": "Abs", "input": ["subtract/sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Abs", "inbound_nodes": [[["subtract", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["tf_op_layer_Abs", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["AUC"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0003499999875202775, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
§"Щ
_tf_keras_input_layer┌{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
§"Щ
_tf_keras_input_layer┌{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
═k
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer-8
layer_with_weights-3
layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
trainable_variables
regularization_losses
 	variables
!	keras_api
+А&call_and_return_all_conditional_losses
б__call__"╝g
_tf_keras_sequentialЮg{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_conv1_input"}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_conv1_input"}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
И
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+Б&call_and_return_all_conditional_losses
ц__call__"Д
_tf_keras_layerЇ{"class_name": "Subtract", "name": "subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128]}, {"class_name": "TensorShape", "items": [null, 128]}]}
й
&trainable_variables
'regularization_losses
(	variables
)	keras_api
+Ц&call_and_return_all_conditional_losses
д__call__"г
_tf_keras_layerњ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Abs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Abs", "trainable": true, "dtype": "float32", "node_def": {"name": "Abs", "op": "Abs", "input": ["subtract/sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Ш

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+Д&call_and_return_all_conditional_losses
е__call__"¤
_tf_keras_layerх{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
Щ
	0decay
1learning_rate
2momentum
3rho
4iter
*rmsљ
+rmsЉ
5rmsњ
6rmsЊ
7rmsћ
8rmsЋ
9rmsќ
:rmsЌ
;rmsў
<rmsЎ
=rmsџ
>rmsЏ
?rmsю
@rmsЮ"
	optimizer
є
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
*12
+13"
trackable_list_wrapper
 "
trackable_list_wrapper
є
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
*12
+13"
trackable_list_wrapper
╬
trainable_variables
Ametrics

Blayers
	regularization_losses
Cnon_trainable_variables
Dlayer_regularization_losses
Elayer_metrics

	variables
а__call__
Ъ_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
Еserving_default"
signature_map
Љ

F_inbound_nodes

5kernel
6bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+ф&call_and_return_all_conditional_losses
Ф__call__"о
_tf_keras_layer╝{"class_name": "Conv2D", "name": "encoder_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
љ
K_inbound_nodes
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"в
_tf_keras_layerЛ{"class_name": "MaxPooling2D", "name": "encoder_pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ѕ
P_inbound_nodes
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+«&call_and_return_all_conditional_losses
»__call__"С
_tf_keras_layer╩{"class_name": "Dropout", "name": "encoder_dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
Љ

U_inbound_nodes

7kernel
8bias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"о
_tf_keras_layer╝{"class_name": "Conv2D", "name": "encoder_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
љ
Z_inbound_nodes
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"в
_tf_keras_layerЛ{"class_name": "MaxPooling2D", "name": "encoder_pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ѕ
__inbound_nodes
`trainable_variables
aregularization_losses
b	variables
c	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"С
_tf_keras_layer╩{"class_name": "Dropout", "name": "encoder_dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
Љ

d_inbound_nodes

9kernel
:bias
etrainable_variables
fregularization_losses
g	variables
h	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"о
_tf_keras_layer╝{"class_name": "Conv2D", "name": "encoder_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
љ
i_inbound_nodes
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"в
_tf_keras_layerЛ{"class_name": "MaxPooling2D", "name": "encoder_pool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ѕ
n_inbound_nodes
otrainable_variables
pregularization_losses
q	variables
r	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"С
_tf_keras_layer╩{"class_name": "Dropout", "name": "encoder_dropout3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
Љ

s_inbound_nodes

;kernel
<bias
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
+╝&call_and_return_all_conditional_losses
й__call__"о
_tf_keras_layer╝{"class_name": "Conv2D", "name": "encoder_conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 48]}}
љ
x_inbound_nodes
ytrainable_variables
zregularization_losses
{	variables
|	keras_api
+Й&call_and_return_all_conditional_losses
┐__call__"в
_tf_keras_layerЛ{"class_name": "MaxPooling2D", "name": "encoder_pool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
І
}_inbound_nodes
~trainable_variables
regularization_losses
ђ	variables
Ђ	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"С
_tf_keras_layer╩{"class_name": "Dropout", "name": "encoder_dropout4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ћ

ѓ_inbound_nodes

=kernel
>bias
Ѓtrainable_variables
ёregularization_losses
Ё	variables
є	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"н
_tf_keras_layer║{"class_name": "Conv2D", "name": "encoder_conv5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
Ћ
Є_inbound_nodes
ѕtrainable_variables
Ѕregularization_losses
і	variables
І	keras_api
+─&call_and_return_all_conditional_losses
┼__call__"в
_tf_keras_layerЛ{"class_name": "MaxPooling2D", "name": "encoder_pool5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ј
ї_inbound_nodes
Їtrainable_variables
јregularization_losses
Ј	variables
љ	keras_api
+к&call_and_return_all_conditional_losses
К__call__"С
_tf_keras_layer╩{"class_name": "Dropout", "name": "encoder_dropout5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
§
Љ_inbound_nodes
њtrainable_variables
Њregularization_losses
ћ	variables
Ћ	keras_api
+╚&call_and_return_all_conditional_losses
╔__call__"М
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ј
ќ_inbound_nodes

?kernel
@bias
Ќtrainable_variables
ўregularization_losses
Ў	variables
џ	keras_api
+╩&call_and_return_all_conditional_losses
╦__call__"¤
_tf_keras_layerх{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1280}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1280]}}
v
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11"
trackable_list_wrapper
х
trainable_variables
Џmetrics
юlayers
regularization_losses
Юnon_trainable_variables
 ъlayer_regularization_losses
Ъlayer_metrics
 	variables
б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
"trainable_variables
аmetrics
Аlayers
#regularization_losses
бnon_trainable_variables
 Бlayer_regularization_losses
цlayer_metrics
$	variables
ц__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
&trainable_variables
Цmetrics
дlayers
'regularization_losses
Дnon_trainable_variables
 еlayer_regularization_losses
Еlayer_metrics
(	variables
д__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
!:	ђ2dense_1/kernel
:2dense_1/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
х
,trainable_variables
фmetrics
Фlayers
-regularization_losses
гnon_trainable_variables
 Гlayer_regularization_losses
«layer_metrics
.	variables
е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
: (2rho
:	 (2RMSprop/iter
9:72sequential/encoder_conv1/kernel
+:)2sequential/encoder_conv1/bias
9:7 2sequential/encoder_conv2/kernel
+:) 2sequential/encoder_conv2/bias
9:7 02sequential/encoder_conv3/kernel
+:)02sequential/encoder_conv3/bias
9:70@2sequential/encoder_conv4/kernel
+:)@2sequential/encoder_conv4/bias
9:7@P2sequential/encoder_conv5/kernel
+:)P2sequential/encoder_conv5/bias
+:)
ђ
ђ2sequential/dense/kernel
$:"ђ2sequential/dense/bias
0
»0
░1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
х
Gtrainable_variables
▒metrics
▓layers
Hregularization_losses
│non_trainable_variables
 ┤layer_regularization_losses
хlayer_metrics
I	variables
Ф__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ltrainable_variables
Хmetrics
иlayers
Mregularization_losses
Иnon_trainable_variables
 ╣layer_regularization_losses
║layer_metrics
N	variables
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Qtrainable_variables
╗metrics
╝layers
Rregularization_losses
йnon_trainable_variables
 Йlayer_regularization_losses
┐layer_metrics
S	variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
х
Vtrainable_variables
└metrics
┴layers
Wregularization_losses
┬non_trainable_variables
 ├layer_regularization_losses
─layer_metrics
X	variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
[trainable_variables
┼metrics
кlayers
\regularization_losses
Кnon_trainable_variables
 ╚layer_regularization_losses
╔layer_metrics
]	variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
`trainable_variables
╩metrics
╦layers
aregularization_losses
╠non_trainable_variables
 ═layer_regularization_losses
╬layer_metrics
b	variables
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
х
etrainable_variables
¤metrics
лlayers
fregularization_losses
Лnon_trainable_variables
 мlayer_regularization_losses
Мlayer_metrics
g	variables
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
jtrainable_variables
нmetrics
Нlayers
kregularization_losses
оnon_trainable_variables
 Оlayer_regularization_losses
пlayer_metrics
l	variables
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
otrainable_variables
┘metrics
┌layers
pregularization_losses
█non_trainable_variables
 ▄layer_regularization_losses
Пlayer_metrics
q	variables
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
х
ttrainable_variables
яmetrics
▀layers
uregularization_losses
Яnon_trainable_variables
 рlayer_regularization_losses
Рlayer_metrics
v	variables
й__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ytrainable_variables
сmetrics
Сlayers
zregularization_losses
тnon_trainable_variables
 Тlayer_regularization_losses
уlayer_metrics
{	variables
┐__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Х
~trainable_variables
Уmetrics
жlayers
regularization_losses
Жnon_trainable_variables
 вlayer_regularization_losses
Вlayer_metrics
ђ	variables
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
И
Ѓtrainable_variables
ьmetrics
Ьlayers
ёregularization_losses
№non_trainable_variables
 ­layer_regularization_losses
ыlayer_metrics
Ё	variables
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѕtrainable_variables
Ыmetrics
зlayers
Ѕregularization_losses
Зnon_trainable_variables
 шlayer_regularization_losses
Шlayer_metrics
і	variables
┼__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Їtrainable_variables
эmetrics
Эlayers
јregularization_losses
щnon_trainable_variables
 Щlayer_regularization_losses
чlayer_metrics
Ј	variables
К__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
њtrainable_variables
Чmetrics
§layers
Њregularization_losses
■non_trainable_variables
  layer_regularization_losses
ђlayer_metrics
ћ	variables
╔__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
И
Ќtrainable_variables
Ђmetrics
ѓlayers
ўregularization_losses
Ѓnon_trainable_variables
 ёlayer_regularization_losses
Ёlayer_metrics
Ў	variables
╦__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
ъ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┐

єtotal

Єcount
ѕ	variables
Ѕ	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
х"
іtrue_positives
Іtrue_negatives
їfalse_positives
Їfalse_negatives
ј	variables
Ј	keras_api"╝!
_tf_keras_metricА!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
є0
Є1"
trackable_list_wrapper
.
ѕ	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
@
і0
І1
ї2
Ї3"
trackable_list_wrapper
.
ј	variables"
_generic_user_object
+:)	ђ2RMSprop/dense_1/kernel/rms
$:"2RMSprop/dense_1/bias/rms
C:A2+RMSprop/sequential/encoder_conv1/kernel/rms
5:32)RMSprop/sequential/encoder_conv1/bias/rms
C:A 2+RMSprop/sequential/encoder_conv2/kernel/rms
5:3 2)RMSprop/sequential/encoder_conv2/bias/rms
C:A 02+RMSprop/sequential/encoder_conv3/kernel/rms
5:302)RMSprop/sequential/encoder_conv3/bias/rms
C:A0@2+RMSprop/sequential/encoder_conv4/kernel/rms
5:3@2)RMSprop/sequential/encoder_conv4/bias/rms
C:A@P2+RMSprop/sequential/encoder_conv5/kernel/rms
5:3P2)RMSprop/sequential/encoder_conv5/bias/rms
5:3
ђ
ђ2#RMSprop/sequential/dense/kernel/rms
.:,ђ2!RMSprop/sequential/dense/bias/rms
Ж2у
G__inference_functional_1_layer_call_and_return_conditional_losses_18100
G__inference_functional_1_layer_call_and_return_conditional_losses_18653
G__inference_functional_1_layer_call_and_return_conditional_losses_18543
G__inference_functional_1_layer_call_and_return_conditional_losses_18150└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
џ2Ќ
 __inference__wrapped_model_17353Ы
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *bб_
]џZ
+і(
input_1         ђђ
+і(
input_2         ђђ
■2ч
,__inference_functional_1_layer_call_fn_18235
,__inference_functional_1_layer_call_fn_18687
,__inference_functional_1_layer_call_fn_18319
,__inference_functional_1_layer_call_fn_18721└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Р2▀
E__inference_sequential_layer_call_and_return_conditional_losses_18814
E__inference_sequential_layer_call_and_return_conditional_losses_17794
E__inference_sequential_layer_call_and_return_conditional_losses_18872
E__inference_sequential_layer_call_and_return_conditional_losses_17749└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ш2з
*__inference_sequential_layer_call_fn_17869
*__inference_sequential_layer_call_fn_18930
*__inference_sequential_layer_call_fn_18901
*__inference_sequential_layer_call_fn_17943└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_subtract_layer_call_and_return_conditional_losses_18936б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_subtract_layer_call_fn_18942б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_18947б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┘2о
/__inference_tf_op_layer_Abs_layer_call_fn_18952б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_1_layer_call_and_return_conditional_losses_18963б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_1_layer_call_fn_18972б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
9B7
#__inference_signature_wrapper_18363input_1input_2
Ы2№
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_18983б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_encoder_conv1_layer_call_fn_18992б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░2Г
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_17359Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ћ2њ
-__inference_encoder_pool1_layer_call_fn_17365Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_19004
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_19009┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
0__inference_encoder_dropout1_layer_call_fn_19014
0__inference_encoder_dropout1_layer_call_fn_19019┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_19030б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_encoder_conv2_layer_call_fn_19039б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░2Г
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_17371Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ћ2њ
-__inference_encoder_pool2_layer_call_fn_17377Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_19051
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_19056┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
0__inference_encoder_dropout2_layer_call_fn_19066
0__inference_encoder_dropout2_layer_call_fn_19061┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_19077б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_encoder_conv3_layer_call_fn_19086б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░2Г
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_17383Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ћ2њ
-__inference_encoder_pool3_layer_call_fn_17389Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_19103
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_19098┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
0__inference_encoder_dropout3_layer_call_fn_19113
0__inference_encoder_dropout3_layer_call_fn_19108┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_19124б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_encoder_conv4_layer_call_fn_19133б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░2Г
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_17395Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ћ2њ
-__inference_encoder_pool4_layer_call_fn_17401Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_19145
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_19150┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
0__inference_encoder_dropout4_layer_call_fn_19155
0__inference_encoder_dropout4_layer_call_fn_19160┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_19171б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_encoder_conv5_layer_call_fn_19180б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░2Г
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_17407Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ћ2њ
-__inference_encoder_pool5_layer_call_fn_17413Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
н2Л
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_19192
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_19197┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
0__inference_encoder_dropout5_layer_call_fn_19207
0__inference_encoder_dropout5_layer_call_fn_19202┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
В2ж
B__inference_flatten_layer_call_and_return_conditional_losses_19213б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_flatten_layer_call_fn_19218б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_19229б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_19238б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 о
 __inference__wrapped_model_17353▒56789:;<=>?@*+lбi
bб_
]џZ
+і(
input_1         ђђ
+і(
input_2         ђђ
ф "1ф.
,
dense_1!і
dense_1         Б
B__inference_dense_1_layer_call_and_return_conditional_losses_18963]*+0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ {
'__inference_dense_1_layer_call_fn_18972P*+0б-
&б#
!і
inputs         ђ
ф "і         б
@__inference_dense_layer_call_and_return_conditional_losses_19229^?@0б-
&б#
!і
inputs         ђ

ф "&б#
і
0         ђ
џ z
%__inference_dense_layer_call_fn_19238Q?@0б-
&б#
!і
inputs         ђ

ф "і         ђ╝
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_18983p569б6
/б,
*і'
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ ћ
-__inference_encoder_conv1_layer_call_fn_18992c569б6
/б,
*і'
inputs         ђђ
ф ""і         ђђИ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_19030l787б4
-б*
(і%
inputs         @@
ф "-б*
#і 
0         @@ 
џ љ
-__inference_encoder_conv2_layer_call_fn_19039_787б4
-б*
(і%
inputs         @@
ф " і         @@ И
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_19077l9:7б4
-б*
(і%
inputs            
ф "-б*
#і 
0           0
џ љ
-__inference_encoder_conv3_layer_call_fn_19086_9:7б4
-б*
(і%
inputs            
ф " і           0И
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_19124l;<7б4
-б*
(і%
inputs         0
ф "-б*
#і 
0         @
џ љ
-__inference_encoder_conv4_layer_call_fn_19133_;<7б4
-б*
(і%
inputs         0
ф " і         @И
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_19171l=>7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         P
џ љ
-__inference_encoder_conv5_layer_call_fn_19180_=>7б4
-б*
(і%
inputs         @
ф " і         P╗
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_19004l;б8
1б.
(і%
inputs         @@
p
ф "-б*
#і 
0         @@
џ ╗
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_19009l;б8
1б.
(і%
inputs         @@
p 
ф "-б*
#і 
0         @@
џ Њ
0__inference_encoder_dropout1_layer_call_fn_19014_;б8
1б.
(і%
inputs         @@
p
ф " і         @@Њ
0__inference_encoder_dropout1_layer_call_fn_19019_;б8
1б.
(і%
inputs         @@
p 
ф " і         @@╗
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_19051l;б8
1б.
(і%
inputs            
p
ф "-б*
#і 
0            
џ ╗
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_19056l;б8
1б.
(і%
inputs            
p 
ф "-б*
#і 
0            
џ Њ
0__inference_encoder_dropout2_layer_call_fn_19061_;б8
1б.
(і%
inputs            
p
ф " і            Њ
0__inference_encoder_dropout2_layer_call_fn_19066_;б8
1б.
(і%
inputs            
p 
ф " і            ╗
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_19098l;б8
1б.
(і%
inputs         0
p
ф "-б*
#і 
0         0
џ ╗
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_19103l;б8
1б.
(і%
inputs         0
p 
ф "-б*
#і 
0         0
џ Њ
0__inference_encoder_dropout3_layer_call_fn_19108_;б8
1б.
(і%
inputs         0
p
ф " і         0Њ
0__inference_encoder_dropout3_layer_call_fn_19113_;б8
1б.
(і%
inputs         0
p 
ф " і         0╗
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_19145l;б8
1б.
(і%
inputs         @
p
ф "-б*
#і 
0         @
џ ╗
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_19150l;б8
1б.
(і%
inputs         @
p 
ф "-б*
#і 
0         @
џ Њ
0__inference_encoder_dropout4_layer_call_fn_19155_;б8
1б.
(і%
inputs         @
p
ф " і         @Њ
0__inference_encoder_dropout4_layer_call_fn_19160_;б8
1б.
(і%
inputs         @
p 
ф " і         @╗
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_19192l;б8
1б.
(і%
inputs         P
p
ф "-б*
#і 
0         P
џ ╗
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_19197l;б8
1б.
(і%
inputs         P
p 
ф "-б*
#і 
0         P
џ Њ
0__inference_encoder_dropout5_layer_call_fn_19202_;б8
1б.
(і%
inputs         P
p
ф " і         PЊ
0__inference_encoder_dropout5_layer_call_fn_19207_;б8
1б.
(і%
inputs         P
p 
ф " і         Pв
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_17359ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_encoder_pool1_layer_call_fn_17365ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    в
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_17371ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_encoder_pool2_layer_call_fn_17377ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    в
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_17383ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_encoder_pool3_layer_call_fn_17389ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    в
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_17395ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_encoder_pool4_layer_call_fn_17401ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    в
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_17407ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_encoder_pool5_layer_call_fn_17413ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Д
B__inference_flatten_layer_call_and_return_conditional_losses_19213a7б4
-б*
(і%
inputs         P
ф "&б#
і
0         ђ

џ 
'__inference_flatten_layer_call_fn_19218T7б4
-б*
(і%
inputs         P
ф "і         ђ
щ
G__inference_functional_1_layer_call_and_return_conditional_losses_18100Г56789:;<=>?@*+tбq
jбg
]џZ
+і(
input_1         ђђ
+і(
input_2         ђђ
p

 
ф "%б"
і
0         
џ щ
G__inference_functional_1_layer_call_and_return_conditional_losses_18150Г56789:;<=>?@*+tбq
jбg
]џZ
+і(
input_1         ђђ
+і(
input_2         ђђ
p 

 
ф "%б"
і
0         
џ ч
G__inference_functional_1_layer_call_and_return_conditional_losses_18543»56789:;<=>?@*+vбs
lбi
_џ\
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ
p

 
ф "%б"
і
0         
џ ч
G__inference_functional_1_layer_call_and_return_conditional_losses_18653»56789:;<=>?@*+vбs
lбi
_џ\
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ
p 

 
ф "%б"
і
0         
џ Л
,__inference_functional_1_layer_call_fn_18235а56789:;<=>?@*+tбq
jбg
]џZ
+і(
input_1         ђђ
+і(
input_2         ђђ
p

 
ф "і         Л
,__inference_functional_1_layer_call_fn_18319а56789:;<=>?@*+tбq
jбg
]џZ
+і(
input_1         ђђ
+і(
input_2         ђђ
p 

 
ф "і         М
,__inference_functional_1_layer_call_fn_18687б56789:;<=>?@*+vбs
lбi
_џ\
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ
p

 
ф "і         М
,__inference_functional_1_layer_call_fn_18721б56789:;<=>?@*+vбs
lбi
_џ\
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ
p 

 
ф "і         л
E__inference_sequential_layer_call_and_return_conditional_losses_17749є56789:;<=>?@NбK
DбA
7і4
encoder_conv1_input         ђђ
p

 
ф "&б#
і
0         ђ
џ л
E__inference_sequential_layer_call_and_return_conditional_losses_17794є56789:;<=>?@NбK
DбA
7і4
encoder_conv1_input         ђђ
p 

 
ф "&б#
і
0         ђ
џ ┬
E__inference_sequential_layer_call_and_return_conditional_losses_18814y56789:;<=>?@Aб>
7б4
*і'
inputs         ђђ
p

 
ф "&б#
і
0         ђ
џ ┬
E__inference_sequential_layer_call_and_return_conditional_losses_18872y56789:;<=>?@Aб>
7б4
*і'
inputs         ђђ
p 

 
ф "&б#
і
0         ђ
џ Д
*__inference_sequential_layer_call_fn_17869y56789:;<=>?@NбK
DбA
7і4
encoder_conv1_input         ђђ
p

 
ф "і         ђД
*__inference_sequential_layer_call_fn_17943y56789:;<=>?@NбK
DбA
7і4
encoder_conv1_input         ђђ
p 

 
ф "і         ђџ
*__inference_sequential_layer_call_fn_18901l56789:;<=>?@Aб>
7б4
*і'
inputs         ђђ
p

 
ф "і         ђџ
*__inference_sequential_layer_call_fn_18930l56789:;<=>?@Aб>
7б4
*і'
inputs         ђђ
p 

 
ф "і         ђЖ
#__inference_signature_wrapper_18363┬56789:;<=>?@*+}бz
б 
sфp
6
input_1+і(
input_1         ђђ
6
input_2+і(
input_2         ђђ"1ф.
,
dense_1!і
dense_1         ╬
C__inference_subtract_layer_call_and_return_conditional_losses_18936є\бY
RбO
MџJ
#і 
inputs/0         ђ
#і 
inputs/1         ђ
ф "&б#
і
0         ђ
џ Ц
(__inference_subtract_layer_call_fn_18942y\бY
RбO
MџJ
#і 
inputs/0         ђ
#і 
inputs/1         ђ
ф "і         ђе
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_18947Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ђ
/__inference_tf_op_layer_Abs_layer_call_fn_18952M0б-
&б#
!і
inputs         ђ
ф "і         ђ