��*
��
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
dtypetype�
�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��#
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
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
�
sequential/encoder_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!sequential/encoder_conv1/kernel
�
3sequential/encoder_conv1/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv1/kernel*&
_output_shapes
:*
dtype0
�
sequential/encoder_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namesequential/encoder_conv1/bias
�
1sequential/encoder_conv1/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv1/bias*
_output_shapes
:*
dtype0
�
sequential/encoder_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!sequential/encoder_conv2/kernel
�
3sequential/encoder_conv2/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv2/kernel*&
_output_shapes
: *
dtype0
�
sequential/encoder_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namesequential/encoder_conv2/bias
�
1sequential/encoder_conv2/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv2/bias*
_output_shapes
: *
dtype0
�
sequential/encoder_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*0
shared_name!sequential/encoder_conv3/kernel
�
3sequential/encoder_conv3/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv3/kernel*&
_output_shapes
: 0*
dtype0
�
sequential/encoder_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_namesequential/encoder_conv3/bias
�
1sequential/encoder_conv3/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv3/bias*
_output_shapes
:0*
dtype0
�
sequential/encoder_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*0
shared_name!sequential/encoder_conv4/kernel
�
3sequential/encoder_conv4/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv4/kernel*&
_output_shapes
:0@*
dtype0
�
sequential/encoder_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namesequential/encoder_conv4/bias
�
1sequential/encoder_conv4/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv4/bias*
_output_shapes
:@*
dtype0
�
sequential/encoder_conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@P*0
shared_name!sequential/encoder_conv5/kernel
�
3sequential/encoder_conv5/kernel/Read/ReadVariableOpReadVariableOpsequential/encoder_conv5/kernel*&
_output_shapes
:@P*
dtype0
�
sequential/encoder_conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*.
shared_namesequential/encoder_conv5/bias
�
1sequential/encoder_conv5/bias/Read/ReadVariableOpReadVariableOpsequential/encoder_conv5/bias*
_output_shapes
:P*
dtype0
�
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�
�*(
shared_namesequential/dense/kernel
�
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel* 
_output_shapes
:
�
�*
dtype0
�
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_namesequential/dense/bias
|
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes	
:�*
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
l

Variable_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
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
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
�
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameRMSprop/dense_1/kernel/rms
�
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes
:	�*
dtype0
�
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_1/bias/rms
�
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:*
dtype0
�
+RMSprop/sequential/encoder_conv1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+RMSprop/sequential/encoder_conv1/kernel/rms
�
?RMSprop/sequential/encoder_conv1/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv1/kernel/rms*&
_output_shapes
:*
dtype0
�
)RMSprop/sequential/encoder_conv1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)RMSprop/sequential/encoder_conv1/bias/rms
�
=RMSprop/sequential/encoder_conv1/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv1/bias/rms*
_output_shapes
:*
dtype0
�
+RMSprop/sequential/encoder_conv2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+RMSprop/sequential/encoder_conv2/kernel/rms
�
?RMSprop/sequential/encoder_conv2/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv2/kernel/rms*&
_output_shapes
: *
dtype0
�
)RMSprop/sequential/encoder_conv2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)RMSprop/sequential/encoder_conv2/bias/rms
�
=RMSprop/sequential/encoder_conv2/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv2/bias/rms*
_output_shapes
: *
dtype0
�
+RMSprop/sequential/encoder_conv3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*<
shared_name-+RMSprop/sequential/encoder_conv3/kernel/rms
�
?RMSprop/sequential/encoder_conv3/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv3/kernel/rms*&
_output_shapes
: 0*
dtype0
�
)RMSprop/sequential/encoder_conv3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*:
shared_name+)RMSprop/sequential/encoder_conv3/bias/rms
�
=RMSprop/sequential/encoder_conv3/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv3/bias/rms*
_output_shapes
:0*
dtype0
�
+RMSprop/sequential/encoder_conv4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*<
shared_name-+RMSprop/sequential/encoder_conv4/kernel/rms
�
?RMSprop/sequential/encoder_conv4/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv4/kernel/rms*&
_output_shapes
:0@*
dtype0
�
)RMSprop/sequential/encoder_conv4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)RMSprop/sequential/encoder_conv4/bias/rms
�
=RMSprop/sequential/encoder_conv4/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv4/bias/rms*
_output_shapes
:@*
dtype0
�
+RMSprop/sequential/encoder_conv5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@P*<
shared_name-+RMSprop/sequential/encoder_conv5/kernel/rms
�
?RMSprop/sequential/encoder_conv5/kernel/rms/Read/ReadVariableOpReadVariableOp+RMSprop/sequential/encoder_conv5/kernel/rms*&
_output_shapes
:@P*
dtype0
�
)RMSprop/sequential/encoder_conv5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*:
shared_name+)RMSprop/sequential/encoder_conv5/bias/rms
�
=RMSprop/sequential/encoder_conv5/bias/rms/Read/ReadVariableOpReadVariableOp)RMSprop/sequential/encoder_conv5/bias/rms*
_output_shapes
:P*
dtype0
�
#RMSprop/sequential/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�
�*4
shared_name%#RMSprop/sequential/dense/kernel/rms
�
7RMSprop/sequential/dense/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/sequential/dense/kernel/rms* 
_output_shapes
:
�
�*
dtype0
�
!RMSprop/sequential/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!RMSprop/sequential/dense/bias/rms
�
5RMSprop/sequential/dense/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/sequential/dense/bias/rms*
_output_shapes	
:�*
dtype0

NoOpNoOp
�p
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�o
value�oB�o B�o
�
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
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
layer-8
layer_with_weights-2
layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
 layer_with_weights-5
 layer-19
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
R
)trainable_variables
*regularization_losses
+	variables
,	keras_api
h

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
�
	3decay
4learning_rate
5momentum
6rho
7iter
-rms�
.rms�
8rms�
9rms�
:rms�
;rms�
<rms�
=rms�
>rms�
?rms�
@rms�
Arms�
Brms�
Crms�
f
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
-12
.13
 
f
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
-12
.13
�
Dmetrics
Elayer_metrics
Fnon_trainable_variables

Glayers
Hlayer_regularization_losses
trainable_variables
	regularization_losses

	variables
 
p
I_rng
J_inbound_nodes
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
p
O_rng
P_inbound_nodes
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
p
U_rng
V_inbound_nodes
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
|
[_inbound_nodes

8kernel
9bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
f
`_inbound_nodes
atrainable_variables
bregularization_losses
c	variables
d	keras_api
f
e_inbound_nodes
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
|
j_inbound_nodes

:kernel
;bias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
f
o_inbound_nodes
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
f
t_inbound_nodes
utrainable_variables
vregularization_losses
w	variables
x	keras_api
|
y_inbound_nodes

<kernel
=bias
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
i
~_inbound_nodes
trainable_variables
�regularization_losses
�	variables
�	keras_api
k
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
�_inbound_nodes

>kernel
?bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
k
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
k
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
�_inbound_nodes

@kernel
Abias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
k
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
k
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
k
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
�_inbound_nodes

Bkernel
Cbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
 
V
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
!trainable_variables
"regularization_losses
#	variables
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
%trainable_variables
&regularization_losses
'	variables
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
)trainable_variables
*regularization_losses
+	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
/trainable_variables
0regularization_losses
1	variables
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
�0
�1
 
 
*
0
1
2
3
4
5
 

�
_state_var
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
Ktrainable_variables
Lregularization_losses
M	variables

�
_state_var
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
Qtrainable_variables
Rregularization_losses
S	variables

�
_state_var
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
Wtrainable_variables
Xregularization_losses
Y	variables
 

80
91
 

80
91
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
\trainable_variables
]regularization_losses
^	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
atrainable_variables
bregularization_losses
c	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ftrainable_variables
gregularization_losses
h	variables
 

:0
;1
 

:0
;1
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ktrainable_variables
lregularization_losses
m	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ptrainable_variables
qregularization_losses
r	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
utrainable_variables
vregularization_losses
w	variables
 

<0
=1
 

<0
=1
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ztrainable_variables
{regularization_losses
|	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
trainable_variables
�regularization_losses
�	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 

>0
?1
 

>0
?1
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 

@0
A1
 

@0
A1
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 
 
 
 
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 

B0
C1
 

B0
C1
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
 
 
 
�
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
17
18
 19
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

�total

�count
�	variables
�	keras_api
v
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api
ec
VARIABLE_VALUEVariableGlayer_with_weights-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ge
VARIABLE_VALUE
Variable_1Glayer_with_weights-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ge
VARIABLE_VALUE
Variable_2Glayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
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
�0
�1

�	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3

�	variables
��
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+RMSprop/sequential/encoder_conv1/kernel/rmsNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)RMSprop/sequential/encoder_conv1/bias/rmsNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+RMSprop/sequential/encoder_conv2/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)RMSprop/sequential/encoder_conv2/bias/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+RMSprop/sequential/encoder_conv3/kernel/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)RMSprop/sequential/encoder_conv3/bias/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+RMSprop/sequential/encoder_conv4/kernel/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)RMSprop/sequential/encoder_conv4/bias/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+RMSprop/sequential/encoder_conv5/kernel/rmsNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)RMSprop/sequential/encoder_conv5/bias/rmsNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#RMSprop/sequential/dense/kernel/rmsOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!RMSprop/sequential/dense/bias/rmsOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
serving_default_input_2Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2sequential/encoder_conv1/kernelsequential/encoder_conv1/biassequential/encoder_conv2/kernelsequential/encoder_conv2/biassequential/encoder_conv3/kernelsequential/encoder_conv3/biassequential/encoder_conv4/kernelsequential/encoder_conv4/biassequential/encoder_conv5/kernelsequential/encoder_conv5/biassequential/dense/kernelsequential/dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_15237
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOprho/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp3sequential/encoder_conv1/kernel/Read/ReadVariableOp1sequential/encoder_conv1/bias/Read/ReadVariableOp3sequential/encoder_conv2/kernel/Read/ReadVariableOp1sequential/encoder_conv2/bias/Read/ReadVariableOp3sequential/encoder_conv3/kernel/Read/ReadVariableOp1sequential/encoder_conv3/bias/Read/ReadVariableOp3sequential/encoder_conv4/kernel/Read/ReadVariableOp1sequential/encoder_conv4/bias/Read/ReadVariableOp3sequential/encoder_conv5/kernel/Read/ReadVariableOp1sequential/encoder_conv5/bias/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv1/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv1/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv2/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv2/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv3/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv3/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv4/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv4/bias/rms/Read/ReadVariableOp?RMSprop/sequential/encoder_conv5/kernel/rms/Read/ReadVariableOp=RMSprop/sequential/encoder_conv5/bias/rms/Read/ReadVariableOp7RMSprop/sequential/dense/kernel/rms/Read/ReadVariableOp5RMSprop/sequential/dense/bias/rms/Read/ReadVariableOpConst*7
Tin0
.2,				*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_17319
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasdecaylearning_ratemomentumrhoRMSprop/itersequential/encoder_conv1/kernelsequential/encoder_conv1/biassequential/encoder_conv2/kernelsequential/encoder_conv2/biassequential/encoder_conv3/kernelsequential/encoder_conv3/biassequential/encoder_conv4/kernelsequential/encoder_conv4/biassequential/encoder_conv5/kernelsequential/encoder_conv5/biassequential/dense/kernelsequential/dense/biasVariable
Variable_1
Variable_2totalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rms+RMSprop/sequential/encoder_conv1/kernel/rms)RMSprop/sequential/encoder_conv1/bias/rms+RMSprop/sequential/encoder_conv2/kernel/rms)RMSprop/sequential/encoder_conv2/bias/rms+RMSprop/sequential/encoder_conv3/kernel/rms)RMSprop/sequential/encoder_conv3/bias/rms+RMSprop/sequential/encoder_conv4/kernel/rms)RMSprop/sequential/encoder_conv4/bias/rms+RMSprop/sequential/encoder_conv5/kernel/rms)RMSprop/sequential/encoder_conv5/bias/rms#RMSprop/sequential/dense/kernel/rms!RMSprop/sequential/dense/bias/rms*6
Tin/
-2+*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_17455�"
�

�
#__inference_signature_wrapper_15237
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
identity��StatefulPartitionedCall�
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
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_137692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
d
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_13935

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_17134

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_145502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�
i
0__inference_encoder_dropout3_layer_call_fn_17024

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_144102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
,__inference_functional_1_layer_call_fn_16019
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

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_150742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes|
z:�����������:�����������::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
I
-__inference_encoder_pool2_layer_call_fn_13917

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_139112
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17108

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������P2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������P2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
-__inference_encoder_conv3_layer_call_fn_17002

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_143812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
I
-__inference_encoder_pool4_layer_call_fn_13941

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_139352
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
I
-__inference_encoder_pool1_layer_call_fn_13905

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_138992
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
-__inference_encoder_conv2_layer_call_fn_16955

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_143232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_14473

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
-__inference_encoder_conv5_layer_call_fn_17096

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_144972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_14381

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������  02
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������  02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   :::W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
L
0__inference_encoder_dropout4_layer_call_fn_17076

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_144732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_14415

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������02

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_14497

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������P2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@:::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_functional_1_layer_call_and_return_conditional_losses_14964
input_1
input_2
sequential_14864
sequential_14866
sequential_14868
sequential_14870
sequential_14872
sequential_14874
sequential_14876
sequential_14878
sequential_14880
sequential_14882
sequential_14884
sequential_14886
sequential_14888
sequential_14890
dense_1_14958
dense_1_14960
identity��dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential/StatefulPartitionedCall_1�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_14864sequential_14866sequential_14868sequential_14870sequential_14872sequential_14874sequential_14876sequential_14878sequential_14880sequential_14882sequential_14884sequential_14886sequential_14888sequential_14890*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_146892$
"sequential/StatefulPartitionedCall�
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_14864sequential_14866sequential_14868sequential_14870sequential_14872sequential_14874sequential_14876sequential_14878sequential_14880sequential_14882sequential_14884sequential_14886sequential_14888sequential_14890#^sequential/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_146892&
$sequential/StatefulPartitionedCall_1�
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_149142
subtract/PartitionedCall�
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_149282!
tf_op_layer_Abs/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_14958dense_1_14960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_149472!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes|
z:�����������:�����������::::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
b
F__inference_random_zoom_layer_call_and_return_conditional_losses_14111

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
q
+__inference_random_zoom_layer_call_fn_16762

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_141072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�\
�
__inference__traced_save_17319
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
0savev2_sequential_dense_bias_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	)
%savev2_variable_2_read_readvariableop	$
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

identity_1��MergeV2Checkpoints�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_950fa8ce7ae04f1a851092e23759829f/part2	
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableopsavev2_rho_read_readvariableop'savev2_rmsprop_iter_read_readvariableop:savev2_sequential_encoder_conv1_kernel_read_readvariableop8savev2_sequential_encoder_conv1_bias_read_readvariableop:savev2_sequential_encoder_conv2_kernel_read_readvariableop8savev2_sequential_encoder_conv2_bias_read_readvariableop:savev2_sequential_encoder_conv3_kernel_read_readvariableop8savev2_sequential_encoder_conv3_bias_read_readvariableop:savev2_sequential_encoder_conv4_kernel_read_readvariableop8savev2_sequential_encoder_conv4_bias_read_readvariableop:savev2_sequential_encoder_conv5_kernel_read_readvariableop8savev2_sequential_encoder_conv5_bias_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv1_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv1_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv2_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv2_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv3_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv3_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv4_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv4_bias_rms_read_readvariableopFsavev2_rmsprop_sequential_encoder_conv5_kernel_rms_read_readvariableopDsavev2_rmsprop_sequential_encoder_conv5_bias_rms_read_readvariableop>savev2_rmsprop_sequential_dense_kernel_rms_read_readvariableop<savev2_rmsprop_sequential_dense_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+				2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�:: : : : : ::: : : 0:0:0@:@:@P:P:
�
�:�:::: : :�:�:�:�:	�:::: : : 0:0:0@:@:@P:P:
�
�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 
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
:: 	

_output_shapes
::,
(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@:,(
&
_output_shapes
:@P: 

_output_shapes
:P:&"
 
_output_shapes
:
�
�:!

_output_shapes	
:�: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
::,!(
&
_output_shapes
: : "

_output_shapes
: :,#(
&
_output_shapes
: 0: $

_output_shapes
:0:,%(
&
_output_shapes
:0@: &

_output_shapes
:@:,'(
&
_output_shapes
:@P: (

_output_shapes
:P:&)"
 
_output_shapes
:
�
�:!*

_output_shapes	
:�:+

_output_shapes
: 
�
�
,__inference_functional_1_layer_call_fn_16053
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
identity��StatefulPartitionedCall�
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
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_151622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
�
-__inference_encoder_conv4_layer_call_fn_17049

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_144392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17019

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������02

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
L
0__inference_encoder_dropout1_layer_call_fn_16935

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_142992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
m
C__inference_subtract_layer_call_and_return_conditional_losses_14914

inputs
inputs_1
identityV
subSubinputsinputs_1*
T0*(
_output_shapes
:����������2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
o
C__inference_subtract_layer_call_and_return_conditional_losses_16500
inputs_0
inputs_1
identityX
subSubinputs_0inputs_1*
T0*(
_output_shapes
:����������2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
L
0__inference_encoder_dropout2_layer_call_fn_16982

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_143572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_17087

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������P2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@:::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_14357

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������   2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������   2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
-__inference_encoder_conv1_layer_call_fn_16908

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_142652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_14526

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������P2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������P*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������P2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������P2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������P2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�B
�
E__inference_sequential_layer_call_and_return_conditional_losses_16432

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
identity��
#encoder_conv1/Conv2D/ReadVariableOpReadVariableOp,encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#encoder_conv1/Conv2D/ReadVariableOp�
encoder_conv1/Conv2DConv2Dinputs+encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
encoder_conv1/Conv2D�
$encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder_conv1/BiasAdd/ReadVariableOp�
encoder_conv1/BiasAddBiasAddencoder_conv1/Conv2D:output:0,encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
encoder_conv1/BiasAdd�
encoder_conv1/EluEluencoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
encoder_conv1/Elu�
encoder_pool1/MaxPoolMaxPoolencoder_conv1/Elu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2
encoder_pool1/MaxPool�
encoder_dropout1/IdentityIdentityencoder_pool1/MaxPool:output:0*
T0*/
_output_shapes
:���������@@2
encoder_dropout1/Identity�
#encoder_conv2/Conv2D/ReadVariableOpReadVariableOp,encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#encoder_conv2/Conv2D/ReadVariableOp�
encoder_conv2/Conv2DConv2D"encoder_dropout1/Identity:output:0+encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
encoder_conv2/Conv2D�
$encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$encoder_conv2/BiasAdd/ReadVariableOp�
encoder_conv2/BiasAddBiasAddencoder_conv2/Conv2D:output:0,encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
encoder_conv2/BiasAdd�
encoder_conv2/EluEluencoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
encoder_conv2/Elu�
encoder_pool2/MaxPoolMaxPoolencoder_conv2/Elu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2
encoder_pool2/MaxPool�
encoder_dropout2/IdentityIdentityencoder_pool2/MaxPool:output:0*
T0*/
_output_shapes
:���������   2
encoder_dropout2/Identity�
#encoder_conv3/Conv2D/ReadVariableOpReadVariableOp,encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02%
#encoder_conv3/Conv2D/ReadVariableOp�
encoder_conv3/Conv2DConv2D"encoder_dropout2/Identity:output:0+encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2
encoder_conv3/Conv2D�
$encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$encoder_conv3/BiasAdd/ReadVariableOp�
encoder_conv3/BiasAddBiasAddencoder_conv3/Conv2D:output:0,encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02
encoder_conv3/BiasAdd�
encoder_conv3/EluEluencoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:���������  02
encoder_conv3/Elu�
encoder_pool3/MaxPoolMaxPoolencoder_conv3/Elu:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2
encoder_pool3/MaxPool�
encoder_dropout3/IdentityIdentityencoder_pool3/MaxPool:output:0*
T0*/
_output_shapes
:���������02
encoder_dropout3/Identity�
#encoder_conv4/Conv2D/ReadVariableOpReadVariableOp,encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02%
#encoder_conv4/Conv2D/ReadVariableOp�
encoder_conv4/Conv2DConv2D"encoder_dropout3/Identity:output:0+encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
encoder_conv4/Conv2D�
$encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$encoder_conv4/BiasAdd/ReadVariableOp�
encoder_conv4/BiasAddBiasAddencoder_conv4/Conv2D:output:0,encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
encoder_conv4/BiasAdd�
encoder_conv4/EluEluencoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
encoder_conv4/Elu�
encoder_pool4/MaxPoolMaxPoolencoder_conv4/Elu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2
encoder_pool4/MaxPool�
encoder_dropout4/IdentityIdentityencoder_pool4/MaxPool:output:0*
T0*/
_output_shapes
:���������@2
encoder_dropout4/Identity�
#encoder_conv5/Conv2D/ReadVariableOpReadVariableOp,encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02%
#encoder_conv5/Conv2D/ReadVariableOp�
encoder_conv5/Conv2DConv2D"encoder_dropout4/Identity:output:0+encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2
encoder_conv5/Conv2D�
$encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder_conv5/BiasAdd/ReadVariableOp�
encoder_conv5/BiasAddBiasAddencoder_conv5/Conv2D:output:0,encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2
encoder_conv5/BiasAdd�
encoder_conv5/EluEluencoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:���������P2
encoder_conv5/Elu�
encoder_pool5/MaxPoolMaxPoolencoder_conv5/Elu:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2
encoder_pool5/MaxPool�
encoder_dropout5/IdentityIdentityencoder_pool5/MaxPool:output:0*
T0*/
_output_shapes
:���������P2
encoder_dropout5/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshape"encoder_dropout5/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������
2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense/Sigmoidf
IdentityIdentitydense/Sigmoid:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������:::::::::::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
߉
�
M__inference_random_translation_layer_call_and_return_conditional_losses_16872

inputs-
)stateful_uniform_statefuluniform_resource
identity�� stateful_uniform/StatefulUniform�"stateful_uniform_1/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1�
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
stateful_uniform/max�
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm�
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype02"
 stateful_uniform/StatefulUniform�
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub�
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform/mul�
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniformc
mulMulstateful_uniform:z:0Cast:y:0*
T0*'
_output_shapes
:���������2
mulz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1�
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform_1/shapeu
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
stateful_uniform_1/max�
,stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2.
,stateful_uniform_1/StatefulUniform/algorithm�
"stateful_uniform_1/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource5stateful_uniform_1/StatefulUniform/algorithm:output:0!stateful_uniform_1/shape:output:0!^stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02$
"stateful_uniform_1/StatefulUniform�
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub�
stateful_uniform_1/mulMul+stateful_uniform_1/StatefulUniform:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1/mul�
stateful_uniform_1Addstateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1k
mul_1Mulstateful_uniform_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������2
mul_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2	mul_1:z:0mul:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concats
translation_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
translation_matrix/Shape�
&translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&translation_matrix/strided_slice/stack�
(translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(translation_matrix/strided_slice/stack_1�
(translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(translation_matrix/strided_slice/stack_2�
 translation_matrix/strided_sliceStridedSlice!translation_matrix/Shape:output:0/translation_matrix/strided_slice/stack:output:01translation_matrix/strided_slice/stack_1:output:01translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 translation_matrix/strided_slice�
translation_matrix/ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
translation_matrix/ones/mul/y�
translation_matrix/ones/mulMul)translation_matrix/strided_slice:output:0&translation_matrix/ones/mul/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/ones/mul�
translation_matrix/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
translation_matrix/ones/Less/y�
translation_matrix/ones/LessLesstranslation_matrix/ones/mul:z:0'translation_matrix/ones/Less/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/ones/Less�
 translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 translation_matrix/ones/packed/1�
translation_matrix/ones/packedPack)translation_matrix/strided_slice:output:0)translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:2 
translation_matrix/ones/packed�
translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
translation_matrix/ones/Const�
translation_matrix/onesFill'translation_matrix/ones/packed:output:0&translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/ones�
translation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
translation_matrix/zeros/mul/y�
translation_matrix/zeros/mulMul)translation_matrix/strided_slice:output:0'translation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/zeros/mul�
translation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
translation_matrix/zeros/Less/y�
translation_matrix/zeros/LessLess translation_matrix/zeros/mul:z:0(translation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/zeros/Less�
!translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!translation_matrix/zeros/packed/1�
translation_matrix/zeros/packedPack)translation_matrix/strided_slice:output:0*translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
translation_matrix/zeros/packed�
translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
translation_matrix/zeros/Const�
translation_matrix/zerosFill(translation_matrix/zeros/packed:output:0'translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/zeros�
(translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(translation_matrix/strided_slice_1/stack�
*translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*translation_matrix/strided_slice_1/stack_1�
*translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*translation_matrix/strided_slice_1/stack_2�
"translation_matrix/strided_slice_1StridedSliceconcat:output:01translation_matrix/strided_slice_1/stack:output:03translation_matrix/strided_slice_1/stack_1:output:03translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2$
"translation_matrix/strided_slice_1�
translation_matrix/NegNeg+translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/Neg�
 translation_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 translation_matrix/zeros_1/mul/y�
translation_matrix/zeros_1/mulMul)translation_matrix/strided_slice:output:0)translation_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2 
translation_matrix/zeros_1/mul�
!translation_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!translation_matrix/zeros_1/Less/y�
translation_matrix/zeros_1/LessLess"translation_matrix/zeros_1/mul:z:0*translation_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2!
translation_matrix/zeros_1/Less�
#translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#translation_matrix/zeros_1/packed/1�
!translation_matrix/zeros_1/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!translation_matrix/zeros_1/packed�
 translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 translation_matrix/zeros_1/Const�
translation_matrix/zeros_1Fill*translation_matrix/zeros_1/packed:output:0)translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/zeros_1�
translation_matrix/ones_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2!
translation_matrix/ones_1/mul/y�
translation_matrix/ones_1/mulMul)translation_matrix/strided_slice:output:0(translation_matrix/ones_1/mul/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/ones_1/mul�
 translation_matrix/ones_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2"
 translation_matrix/ones_1/Less/y�
translation_matrix/ones_1/LessLess!translation_matrix/ones_1/mul:z:0)translation_matrix/ones_1/Less/y:output:0*
T0*
_output_shapes
: 2 
translation_matrix/ones_1/Less�
"translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"translation_matrix/ones_1/packed/1�
 translation_matrix/ones_1/packedPack)translation_matrix/strided_slice:output:0+translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 translation_matrix/ones_1/packed�
translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
translation_matrix/ones_1/Const�
translation_matrix/ones_1Fill)translation_matrix/ones_1/packed:output:0(translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/ones_1�
(translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(translation_matrix/strided_slice_2/stack�
*translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*translation_matrix/strided_slice_2/stack_1�
*translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*translation_matrix/strided_slice_2/stack_2�
"translation_matrix/strided_slice_2StridedSliceconcat:output:01translation_matrix/strided_slice_2/stack:output:03translation_matrix/strided_slice_2/stack_1:output:03translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2$
"translation_matrix/strided_slice_2�
translation_matrix/Neg_1Neg+translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/Neg_1�
 translation_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 translation_matrix/zeros_2/mul/y�
translation_matrix/zeros_2/mulMul)translation_matrix/strided_slice:output:0)translation_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2 
translation_matrix/zeros_2/mul�
!translation_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!translation_matrix/zeros_2/Less/y�
translation_matrix/zeros_2/LessLess"translation_matrix/zeros_2/mul:z:0*translation_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2!
translation_matrix/zeros_2/Less�
#translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#translation_matrix/zeros_2/packed/1�
!translation_matrix/zeros_2/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!translation_matrix/zeros_2/packed�
 translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 translation_matrix/zeros_2/Const�
translation_matrix/zeros_2Fill*translation_matrix/zeros_2/packed:output:0)translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/zeros_2�
translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
translation_matrix/concat/axis�
translation_matrix/concatConcatV2 translation_matrix/ones:output:0!translation_matrix/zeros:output:0translation_matrix/Neg:y:0#translation_matrix/zeros_1:output:0"translation_matrix/ones_1:output:0translation_matrix/Neg_1:y:0#translation_matrix/zeros_2:output:0'translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
translation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape�
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack�
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1�
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice�
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputs"translation_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2�
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform#^stateful_uniform_1/StatefulUniform*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform2H
"stateful_uniform_1/StatefulUniform"stateful_uniform_1/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_16494

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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_147702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_16925

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
K
/__inference_tf_op_layer_Abs_layer_call_fn_16516

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_149282
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�(
e
F__inference_random_flip_layer_call_and_return_conditional_losses_16630

inputs
identity��
)random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:�����������2+
)random_flip_left_right/control_dependency�
random_flip_left_right/ShapeShape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2
random_flip_left_right/Shape�
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*random_flip_left_right/strided_slice/stack�
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_1�
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2�
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$random_flip_left_right/strided_slice�
+random_flip_left_right/random_uniform/shapePack-random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape�
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min�
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)random_flip_left_right/random_uniform/max�
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype025
3random_flip_left_right/random_uniform/RandomUniform�
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:���������2+
)random_flip_left_right/random_uniform/Mul�
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1�
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2�
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3�
$random_flip_left_right/Reshape/shapePack-random_flip_left_right/strided_slice:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shape�
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2 
random_flip_left_right/Reshape�
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/Round�
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axis�
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:�����������2"
 random_flip_left_right/ReverseV2�
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:�����������2
random_flip_left_right/mul�
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_flip_left_right/sub/x�
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/sub�
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:�����������2
random_flip_left_right/mul_1�
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:�����������2
random_flip_left_right/add|
IdentityIdentityrandom_flip_left_right/add:z:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�]
�
E__inference_sequential_layer_call_and_return_conditional_losses_14586
random_flip_input
random_zoom_14125
random_translation_14251
encoder_conv1_14276
encoder_conv1_14278
encoder_conv2_14334
encoder_conv2_14336
encoder_conv3_14392
encoder_conv3_14394
encoder_conv4_14450
encoder_conv4_14452
encoder_conv5_14508
encoder_conv5_14510
dense_14580
dense_14582
identity��dense/StatefulPartitionedCall�%encoder_conv1/StatefulPartitionedCall�%encoder_conv2/StatefulPartitionedCall�%encoder_conv3/StatefulPartitionedCall�%encoder_conv4/StatefulPartitionedCall�%encoder_conv5/StatefulPartitionedCall�(encoder_dropout1/StatefulPartitionedCall�(encoder_dropout2/StatefulPartitionedCall�(encoder_dropout3/StatefulPartitionedCall�(encoder_dropout4/StatefulPartitionedCall�(encoder_dropout5/StatefulPartitionedCall�#random_flip/StatefulPartitionedCall�*random_translation/StatefulPartitionedCall�#random_zoom/StatefulPartitionedCall�
#random_flip/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_139842%
#random_flip/StatefulPartitionedCall�
#random_zoom/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_zoom_14125*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_141072%
#random_zoom/StatefulPartitionedCall�
*random_translation/StatefulPartitionedCallStatefulPartitionedCall,random_zoom/StatefulPartitionedCall:output:0random_translation_14251*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_translation_layer_call_and_return_conditional_losses_142332,
*random_translation/StatefulPartitionedCall�
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCall3random_translation/StatefulPartitionedCall:output:0encoder_conv1_14276encoder_conv1_14278*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_142652'
%encoder_conv1/StatefulPartitionedCall�
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_138992
encoder_pool1/PartitionedCall�
(encoder_dropout1/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool1/PartitionedCall:output:0$^random_flip/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_142942*
(encoder_dropout1/StatefulPartitionedCall�
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout1/StatefulPartitionedCall:output:0encoder_conv2_14334encoder_conv2_14336*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_143232'
%encoder_conv2/StatefulPartitionedCall�
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_139112
encoder_pool2/PartitionedCall�
(encoder_dropout2/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool2/PartitionedCall:output:0)^encoder_dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_143522*
(encoder_dropout2/StatefulPartitionedCall�
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout2/StatefulPartitionedCall:output:0encoder_conv3_14392encoder_conv3_14394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_143812'
%encoder_conv3/StatefulPartitionedCall�
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_139232
encoder_pool3/PartitionedCall�
(encoder_dropout3/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool3/PartitionedCall:output:0)^encoder_dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_144102*
(encoder_dropout3/StatefulPartitionedCall�
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout3/StatefulPartitionedCall:output:0encoder_conv4_14450encoder_conv4_14452*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_144392'
%encoder_conv4/StatefulPartitionedCall�
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_139352
encoder_pool4/PartitionedCall�
(encoder_dropout4/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool4/PartitionedCall:output:0)^encoder_dropout3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_144682*
(encoder_dropout4/StatefulPartitionedCall�
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout4/StatefulPartitionedCall:output:0encoder_conv5_14508encoder_conv5_14510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_144972'
%encoder_conv5/StatefulPartitionedCall�
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_139472
encoder_pool5/PartitionedCall�
(encoder_dropout5/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool5/PartitionedCall:output:0)^encoder_dropout4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_145262*
(encoder_dropout5/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall1encoder_dropout5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_145502
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14580dense_14582*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_145692
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall)^encoder_dropout1/StatefulPartitionedCall)^encoder_dropout2/StatefulPartitionedCall)^encoder_dropout3/StatefulPartitionedCall)^encoder_dropout4/StatefulPartitionedCall)^encoder_dropout5/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall+^random_translation/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:�����������::::::::::::::2>
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
(encoder_dropout5/StatefulPartitionedCall(encoder_dropout5/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2X
*random_translation/StatefulPartitionedCall*random_translation/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:d `
1
_output_shapes
:�����������
+
_user_specified_namerandom_flip_input
�	
�
*__inference_sequential_layer_call_fn_14797
random_flip_input
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_147702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:�����������
+
_user_specified_namerandom_flip_input
�
�
,__inference_functional_1_layer_call_fn_15109
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

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_150742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes|
z:�����������:�����������::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
x
2__inference_random_translation_layer_call_fn_16883

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_translation_layer_call_and_return_conditional_losses_142332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_16899

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Eluo
IdentityIdentityElu:activations:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������:::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_14928

inputs
identity[
AbsAbsinputs*
T0*
_cloned(*(
_output_shapes
:����������2
Abs\
IdentityIdentityAbs:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
߉
�
M__inference_random_translation_layer_call_and_return_conditional_losses_14233

inputs-
)stateful_uniform_statefuluniform_resource
identity�� stateful_uniform/StatefulUniform�"stateful_uniform_1/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1�
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
stateful_uniform/max�
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm�
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype02"
 stateful_uniform/StatefulUniform�
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub�
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform/mul�
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniformc
mulMulstateful_uniform:z:0Cast:y:0*
T0*'
_output_shapes
:���������2
mulz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1�
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform_1/shapeu
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
stateful_uniform_1/max�
,stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2.
,stateful_uniform_1/StatefulUniform/algorithm�
"stateful_uniform_1/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource5stateful_uniform_1/StatefulUniform/algorithm:output:0!stateful_uniform_1/shape:output:0!^stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02$
"stateful_uniform_1/StatefulUniform�
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub�
stateful_uniform_1/mulMul+stateful_uniform_1/StatefulUniform:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1/mul�
stateful_uniform_1Addstateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1k
mul_1Mulstateful_uniform_1:z:0
Cast_1:y:0*
T0*'
_output_shapes
:���������2
mul_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2	mul_1:z:0mul:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concats
translation_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
translation_matrix/Shape�
&translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&translation_matrix/strided_slice/stack�
(translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(translation_matrix/strided_slice/stack_1�
(translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(translation_matrix/strided_slice/stack_2�
 translation_matrix/strided_sliceStridedSlice!translation_matrix/Shape:output:0/translation_matrix/strided_slice/stack:output:01translation_matrix/strided_slice/stack_1:output:01translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 translation_matrix/strided_slice�
translation_matrix/ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
translation_matrix/ones/mul/y�
translation_matrix/ones/mulMul)translation_matrix/strided_slice:output:0&translation_matrix/ones/mul/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/ones/mul�
translation_matrix/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
translation_matrix/ones/Less/y�
translation_matrix/ones/LessLesstranslation_matrix/ones/mul:z:0'translation_matrix/ones/Less/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/ones/Less�
 translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 translation_matrix/ones/packed/1�
translation_matrix/ones/packedPack)translation_matrix/strided_slice:output:0)translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:2 
translation_matrix/ones/packed�
translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
translation_matrix/ones/Const�
translation_matrix/onesFill'translation_matrix/ones/packed:output:0&translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/ones�
translation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
translation_matrix/zeros/mul/y�
translation_matrix/zeros/mulMul)translation_matrix/strided_slice:output:0'translation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/zeros/mul�
translation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
translation_matrix/zeros/Less/y�
translation_matrix/zeros/LessLess translation_matrix/zeros/mul:z:0(translation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/zeros/Less�
!translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!translation_matrix/zeros/packed/1�
translation_matrix/zeros/packedPack)translation_matrix/strided_slice:output:0*translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
translation_matrix/zeros/packed�
translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
translation_matrix/zeros/Const�
translation_matrix/zerosFill(translation_matrix/zeros/packed:output:0'translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/zeros�
(translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(translation_matrix/strided_slice_1/stack�
*translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*translation_matrix/strided_slice_1/stack_1�
*translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*translation_matrix/strided_slice_1/stack_2�
"translation_matrix/strided_slice_1StridedSliceconcat:output:01translation_matrix/strided_slice_1/stack:output:03translation_matrix/strided_slice_1/stack_1:output:03translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2$
"translation_matrix/strided_slice_1�
translation_matrix/NegNeg+translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/Neg�
 translation_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 translation_matrix/zeros_1/mul/y�
translation_matrix/zeros_1/mulMul)translation_matrix/strided_slice:output:0)translation_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2 
translation_matrix/zeros_1/mul�
!translation_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!translation_matrix/zeros_1/Less/y�
translation_matrix/zeros_1/LessLess"translation_matrix/zeros_1/mul:z:0*translation_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2!
translation_matrix/zeros_1/Less�
#translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#translation_matrix/zeros_1/packed/1�
!translation_matrix/zeros_1/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!translation_matrix/zeros_1/packed�
 translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 translation_matrix/zeros_1/Const�
translation_matrix/zeros_1Fill*translation_matrix/zeros_1/packed:output:0)translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/zeros_1�
translation_matrix/ones_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2!
translation_matrix/ones_1/mul/y�
translation_matrix/ones_1/mulMul)translation_matrix/strided_slice:output:0(translation_matrix/ones_1/mul/y:output:0*
T0*
_output_shapes
: 2
translation_matrix/ones_1/mul�
 translation_matrix/ones_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2"
 translation_matrix/ones_1/Less/y�
translation_matrix/ones_1/LessLess!translation_matrix/ones_1/mul:z:0)translation_matrix/ones_1/Less/y:output:0*
T0*
_output_shapes
: 2 
translation_matrix/ones_1/Less�
"translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"translation_matrix/ones_1/packed/1�
 translation_matrix/ones_1/packedPack)translation_matrix/strided_slice:output:0+translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 translation_matrix/ones_1/packed�
translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
translation_matrix/ones_1/Const�
translation_matrix/ones_1Fill)translation_matrix/ones_1/packed:output:0(translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/ones_1�
(translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(translation_matrix/strided_slice_2/stack�
*translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*translation_matrix/strided_slice_2/stack_1�
*translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*translation_matrix/strided_slice_2/stack_2�
"translation_matrix/strided_slice_2StridedSliceconcat:output:01translation_matrix/strided_slice_2/stack:output:03translation_matrix/strided_slice_2/stack_1:output:03translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2$
"translation_matrix/strided_slice_2�
translation_matrix/Neg_1Neg+translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/Neg_1�
 translation_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 translation_matrix/zeros_2/mul/y�
translation_matrix/zeros_2/mulMul)translation_matrix/strided_slice:output:0)translation_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2 
translation_matrix/zeros_2/mul�
!translation_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!translation_matrix/zeros_2/Less/y�
translation_matrix/zeros_2/LessLess"translation_matrix/zeros_2/mul:z:0*translation_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2!
translation_matrix/zeros_2/Less�
#translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#translation_matrix/zeros_2/packed/1�
!translation_matrix/zeros_2/packedPack)translation_matrix/strided_slice:output:0,translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!translation_matrix/zeros_2/packed�
 translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 translation_matrix/zeros_2/Const�
translation_matrix/zeros_2Fill*translation_matrix/zeros_2/packed:output:0)translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2
translation_matrix/zeros_2�
translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
translation_matrix/concat/axis�
translation_matrix/concatConcatV2 translation_matrix/ones:output:0!translation_matrix/zeros:output:0translation_matrix/Neg:y:0#translation_matrix/zeros_1:output:0"translation_matrix/ones_1:output:0translation_matrix/Neg_1:y:0#translation_matrix/zeros_2:output:0'translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
translation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape�
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack�
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1�
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice�
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputs"translation_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2�
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform#^stateful_uniform_1/StatefulUniform*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform2H
"stateful_uniform_1/StatefulUniform"stateful_uniform_1/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�`
�
F__inference_random_flip_layer_call_and_return_conditional_losses_13881

inputs
identity��9random_flip_left_right/assert_greater_equal/Assert/Assert�@random_flip_left_right/assert_positive/assert_less/Assert/Assertr
random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2
random_flip_left_right/Shape�
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*random_flip_left_right/strided_slice/stack�
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice/stack_1�
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2�
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$random_flip_left_right/strided_slice�
,random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,random_flip_left_right/assert_positive/Const�
7random_flip_left_right/assert_positive/assert_less/LessLess5random_flip_left_right/assert_positive/Const:output:0-random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:29
7random_flip_left_right/assert_positive/assert_less/Less�
8random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_left_right/assert_positive/assert_less/Const�
6random_flip_left_right/assert_positive/assert_less/AllAll;random_flip_left_right/assert_positive/assert_less/Less:z:0Arandom_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 28
6random_flip_left_right/assert_positive/assert_less/All�
?random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2A
?random_flip_left_right/assert_positive/assert_less/Assert/Const�
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2I
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0�
@random_flip_left_right/assert_positive/assert_less/Assert/AssertAssert?random_flip_left_right/assert_positive/assert_less/All:output:0Prandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2B
@random_flip_left_right/assert_positive/assert_less/Assert/Assert|
random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2
random_flip_left_right/Rank�
-random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_flip_left_right/assert_greater_equal/y�
8random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual$random_flip_left_right/Rank:output:06random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2:
8random_flip_left_right/assert_greater_equal/GreaterEqual�
1random_flip_left_right/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 23
1random_flip_left_right/assert_greater_equal/Const�
/random_flip_left_right/assert_greater_equal/AllAll<random_flip_left_right/assert_greater_equal/GreaterEqual:z:0:random_flip_left_right/assert_greater_equal/Const:output:0*
_output_shapes
: 21
/random_flip_left_right/assert_greater_equal/All�
8random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2:
8random_flip_left_right/assert_greater_equal/Assert/Const�
:random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2<
:random_flip_left_right/assert_greater_equal/Assert/Const_1�
:random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_2�
:random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_3�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4�
9random_flip_left_right/assert_greater_equal/Assert/AssertAssert8random_flip_left_right/assert_greater_equal/All:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0$random_flip_left_right/Rank:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:06random_flip_left_right/assert_greater_equal/y:output:0A^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2;
9random_flip_left_right/assert_greater_equal/Assert/Assert�
)random_flip_left_right/control_dependencyIdentityinputs:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4������������������������������������2+
)random_flip_left_right/control_dependency�
random_flip_left_right/Shape_1Shape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2 
random_flip_left_right/Shape_1�
,random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice_1/stack�
.random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_1�
.random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_2�
&random_flip_left_right/strided_slice_1StridedSlice'random_flip_left_right/Shape_1:output:05random_flip_left_right/strided_slice_1/stack:output:07random_flip_left_right/strided_slice_1/stack_1:output:07random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&random_flip_left_right/strided_slice_1�
+random_flip_left_right/random_uniform/shapePack/random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape�
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min�
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)random_flip_left_right/random_uniform/max�
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype025
3random_flip_left_right/random_uniform/RandomUniform�
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:���������2+
)random_flip_left_right/random_uniform/Mul�
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1�
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2�
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3�
$random_flip_left_right/Reshape/shapePack/random_flip_left_right/strided_slice_1:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shape�
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2 
random_flip_left_right/Reshape�
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/Round�
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axis�
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4������������������������������������2"
 random_flip_left_right/ReverseV2�
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
random_flip_left_right/mul�
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_flip_left_right/sub/x�
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/sub�
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
random_flip_left_right/mul_1�
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2
random_flip_left_right/add�
IdentityIdentityrandom_flip_left_right/add:z:0:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������2v
9random_flip_left_right/assert_greater_equal/Assert/Assert9random_flip_left_right/assert_greater_equal/Assert/Assert2�
@random_flip_left_right/assert_positive/assert_less/Assert/Assert@random_flip_left_right/assert_positive/assert_less/Assert/Assert:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
0__inference_encoder_dropout5_layer_call_fn_17118

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_145262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_13769
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
identity��
;functional_1/sequential/encoder_conv1/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;functional_1/sequential/encoder_conv1/Conv2D/ReadVariableOp�
,functional_1/sequential/encoder_conv1/Conv2DConv2Dinput_1Cfunctional_1/sequential/encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv1/Conv2D�
<functional_1/sequential/encoder_conv1/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<functional_1/sequential/encoder_conv1/BiasAdd/ReadVariableOp�
-functional_1/sequential/encoder_conv1/BiasAddBiasAdd5functional_1/sequential/encoder_conv1/Conv2D:output:0Dfunctional_1/sequential/encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2/
-functional_1/sequential/encoder_conv1/BiasAdd�
)functional_1/sequential/encoder_conv1/EluElu6functional_1/sequential/encoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2+
)functional_1/sequential/encoder_conv1/Elu�
-functional_1/sequential/encoder_pool1/MaxPoolMaxPool7functional_1/sequential/encoder_conv1/Elu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool1/MaxPool�
1functional_1/sequential/encoder_dropout1/IdentityIdentity6functional_1/sequential/encoder_pool1/MaxPool:output:0*
T0*/
_output_shapes
:���������@@23
1functional_1/sequential/encoder_dropout1/Identity�
;functional_1/sequential/encoder_conv2/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;functional_1/sequential/encoder_conv2/Conv2D/ReadVariableOp�
,functional_1/sequential/encoder_conv2/Conv2DConv2D:functional_1/sequential/encoder_dropout1/Identity:output:0Cfunctional_1/sequential/encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv2/Conv2D�
<functional_1/sequential/encoder_conv2/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<functional_1/sequential/encoder_conv2/BiasAdd/ReadVariableOp�
-functional_1/sequential/encoder_conv2/BiasAddBiasAdd5functional_1/sequential/encoder_conv2/Conv2D:output:0Dfunctional_1/sequential/encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2/
-functional_1/sequential/encoder_conv2/BiasAdd�
)functional_1/sequential/encoder_conv2/EluElu6functional_1/sequential/encoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2+
)functional_1/sequential/encoder_conv2/Elu�
-functional_1/sequential/encoder_pool2/MaxPoolMaxPool7functional_1/sequential/encoder_conv2/Elu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool2/MaxPool�
1functional_1/sequential/encoder_dropout2/IdentityIdentity6functional_1/sequential/encoder_pool2/MaxPool:output:0*
T0*/
_output_shapes
:���������   23
1functional_1/sequential/encoder_dropout2/Identity�
;functional_1/sequential/encoder_conv3/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;functional_1/sequential/encoder_conv3/Conv2D/ReadVariableOp�
,functional_1/sequential/encoder_conv3/Conv2DConv2D:functional_1/sequential/encoder_dropout2/Identity:output:0Cfunctional_1/sequential/encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv3/Conv2D�
<functional_1/sequential/encoder_conv3/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02>
<functional_1/sequential/encoder_conv3/BiasAdd/ReadVariableOp�
-functional_1/sequential/encoder_conv3/BiasAddBiasAdd5functional_1/sequential/encoder_conv3/Conv2D:output:0Dfunctional_1/sequential/encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02/
-functional_1/sequential/encoder_conv3/BiasAdd�
)functional_1/sequential/encoder_conv3/EluElu6functional_1/sequential/encoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:���������  02+
)functional_1/sequential/encoder_conv3/Elu�
-functional_1/sequential/encoder_pool3/MaxPoolMaxPool7functional_1/sequential/encoder_conv3/Elu:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool3/MaxPool�
1functional_1/sequential/encoder_dropout3/IdentityIdentity6functional_1/sequential/encoder_pool3/MaxPool:output:0*
T0*/
_output_shapes
:���������023
1functional_1/sequential/encoder_dropout3/Identity�
;functional_1/sequential/encoder_conv4/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02=
;functional_1/sequential/encoder_conv4/Conv2D/ReadVariableOp�
,functional_1/sequential/encoder_conv4/Conv2DConv2D:functional_1/sequential/encoder_dropout3/Identity:output:0Cfunctional_1/sequential/encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv4/Conv2D�
<functional_1/sequential/encoder_conv4/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<functional_1/sequential/encoder_conv4/BiasAdd/ReadVariableOp�
-functional_1/sequential/encoder_conv4/BiasAddBiasAdd5functional_1/sequential/encoder_conv4/Conv2D:output:0Dfunctional_1/sequential/encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2/
-functional_1/sequential/encoder_conv4/BiasAdd�
)functional_1/sequential/encoder_conv4/EluElu6functional_1/sequential/encoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2+
)functional_1/sequential/encoder_conv4/Elu�
-functional_1/sequential/encoder_pool4/MaxPoolMaxPool7functional_1/sequential/encoder_conv4/Elu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool4/MaxPool�
1functional_1/sequential/encoder_dropout4/IdentityIdentity6functional_1/sequential/encoder_pool4/MaxPool:output:0*
T0*/
_output_shapes
:���������@23
1functional_1/sequential/encoder_dropout4/Identity�
;functional_1/sequential/encoder_conv5/Conv2D/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02=
;functional_1/sequential/encoder_conv5/Conv2D/ReadVariableOp�
,functional_1/sequential/encoder_conv5/Conv2DConv2D:functional_1/sequential/encoder_dropout4/Identity:output:0Cfunctional_1/sequential/encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2.
,functional_1/sequential/encoder_conv5/Conv2D�
<functional_1/sequential/encoder_conv5/BiasAdd/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02>
<functional_1/sequential/encoder_conv5/BiasAdd/ReadVariableOp�
-functional_1/sequential/encoder_conv5/BiasAddBiasAdd5functional_1/sequential/encoder_conv5/Conv2D:output:0Dfunctional_1/sequential/encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2/
-functional_1/sequential/encoder_conv5/BiasAdd�
)functional_1/sequential/encoder_conv5/EluElu6functional_1/sequential/encoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:���������P2+
)functional_1/sequential/encoder_conv5/Elu�
-functional_1/sequential/encoder_pool5/MaxPoolMaxPool7functional_1/sequential/encoder_conv5/Elu:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2/
-functional_1/sequential/encoder_pool5/MaxPool�
1functional_1/sequential/encoder_dropout5/IdentityIdentity6functional_1/sequential/encoder_pool5/MaxPool:output:0*
T0*/
_output_shapes
:���������P23
1functional_1/sequential/encoder_dropout5/Identity�
%functional_1/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2'
%functional_1/sequential/flatten/Const�
'functional_1/sequential/flatten/ReshapeReshape:functional_1/sequential/encoder_dropout5/Identity:output:0.functional_1/sequential/flatten/Const:output:0*
T0*(
_output_shapes
:����������
2)
'functional_1/sequential/flatten/Reshape�
3functional_1/sequential/dense/MatMul/ReadVariableOpReadVariableOp<functional_1_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype025
3functional_1/sequential/dense/MatMul/ReadVariableOp�
$functional_1/sequential/dense/MatMulMatMul0functional_1/sequential/flatten/Reshape:output:0;functional_1/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$functional_1/sequential/dense/MatMul�
4functional_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=functional_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4functional_1/sequential/dense/BiasAdd/ReadVariableOp�
%functional_1/sequential/dense/BiasAddBiasAdd.functional_1/sequential/dense/MatMul:product:0<functional_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2'
%functional_1/sequential/dense/BiasAdd�
%functional_1/sequential/dense/SigmoidSigmoid.functional_1/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2'
%functional_1/sequential/dense/Sigmoid�
=functional_1/sequential/encoder_conv1/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02?
=functional_1/sequential/encoder_conv1/Conv2D_1/ReadVariableOp�
.functional_1/sequential/encoder_conv1/Conv2D_1Conv2Dinput_2Efunctional_1/sequential/encoder_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv1/Conv2D_1�
>functional_1/sequential/encoder_conv1/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>functional_1/sequential/encoder_conv1/BiasAdd_1/ReadVariableOp�
/functional_1/sequential/encoder_conv1/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv1/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������21
/functional_1/sequential/encoder_conv1/BiasAdd_1�
+functional_1/sequential/encoder_conv1/Elu_1Elu8functional_1/sequential/encoder_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:�����������2-
+functional_1/sequential/encoder_conv1/Elu_1�
/functional_1/sequential/encoder_pool1/MaxPool_1MaxPool9functional_1/sequential/encoder_conv1/Elu_1:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool1/MaxPool_1�
3functional_1/sequential/encoder_dropout1/Identity_1Identity8functional_1/sequential/encoder_pool1/MaxPool_1:output:0*
T0*/
_output_shapes
:���������@@25
3functional_1/sequential/encoder_dropout1/Identity_1�
=functional_1/sequential/encoder_conv2/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=functional_1/sequential/encoder_conv2/Conv2D_1/ReadVariableOp�
.functional_1/sequential/encoder_conv2/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout1/Identity_1:output:0Efunctional_1/sequential/encoder_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv2/Conv2D_1�
>functional_1/sequential/encoder_conv2/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>functional_1/sequential/encoder_conv2/BiasAdd_1/ReadVariableOp�
/functional_1/sequential/encoder_conv2/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv2/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 21
/functional_1/sequential/encoder_conv2/BiasAdd_1�
+functional_1/sequential/encoder_conv2/Elu_1Elu8functional_1/sequential/encoder_conv2/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������@@ 2-
+functional_1/sequential/encoder_conv2/Elu_1�
/functional_1/sequential/encoder_pool2/MaxPool_1MaxPool9functional_1/sequential/encoder_conv2/Elu_1:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool2/MaxPool_1�
3functional_1/sequential/encoder_dropout2/Identity_1Identity8functional_1/sequential/encoder_pool2/MaxPool_1:output:0*
T0*/
_output_shapes
:���������   25
3functional_1/sequential/encoder_dropout2/Identity_1�
=functional_1/sequential/encoder_conv3/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02?
=functional_1/sequential/encoder_conv3/Conv2D_1/ReadVariableOp�
.functional_1/sequential/encoder_conv3/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout2/Identity_1:output:0Efunctional_1/sequential/encoder_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv3/Conv2D_1�
>functional_1/sequential/encoder_conv3/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02@
>functional_1/sequential/encoder_conv3/BiasAdd_1/ReadVariableOp�
/functional_1/sequential/encoder_conv3/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv3/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  021
/functional_1/sequential/encoder_conv3/BiasAdd_1�
+functional_1/sequential/encoder_conv3/Elu_1Elu8functional_1/sequential/encoder_conv3/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������  02-
+functional_1/sequential/encoder_conv3/Elu_1�
/functional_1/sequential/encoder_pool3/MaxPool_1MaxPool9functional_1/sequential/encoder_conv3/Elu_1:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool3/MaxPool_1�
3functional_1/sequential/encoder_dropout3/Identity_1Identity8functional_1/sequential/encoder_pool3/MaxPool_1:output:0*
T0*/
_output_shapes
:���������025
3functional_1/sequential/encoder_dropout3/Identity_1�
=functional_1/sequential/encoder_conv4/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02?
=functional_1/sequential/encoder_conv4/Conv2D_1/ReadVariableOp�
.functional_1/sequential/encoder_conv4/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout3/Identity_1:output:0Efunctional_1/sequential/encoder_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv4/Conv2D_1�
>functional_1/sequential/encoder_conv4/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02@
>functional_1/sequential/encoder_conv4/BiasAdd_1/ReadVariableOp�
/functional_1/sequential/encoder_conv4/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv4/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@21
/functional_1/sequential/encoder_conv4/BiasAdd_1�
+functional_1/sequential/encoder_conv4/Elu_1Elu8functional_1/sequential/encoder_conv4/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������@2-
+functional_1/sequential/encoder_conv4/Elu_1�
/functional_1/sequential/encoder_pool4/MaxPool_1MaxPool9functional_1/sequential/encoder_conv4/Elu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool4/MaxPool_1�
3functional_1/sequential/encoder_dropout4/Identity_1Identity8functional_1/sequential/encoder_pool4/MaxPool_1:output:0*
T0*/
_output_shapes
:���������@25
3functional_1/sequential/encoder_dropout4/Identity_1�
=functional_1/sequential/encoder_conv5/Conv2D_1/ReadVariableOpReadVariableOpDfunctional_1_sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02?
=functional_1/sequential/encoder_conv5/Conv2D_1/ReadVariableOp�
.functional_1/sequential/encoder_conv5/Conv2D_1Conv2D<functional_1/sequential/encoder_dropout4/Identity_1:output:0Efunctional_1/sequential/encoder_conv5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
20
.functional_1/sequential/encoder_conv5/Conv2D_1�
>functional_1/sequential/encoder_conv5/BiasAdd_1/ReadVariableOpReadVariableOpEfunctional_1_sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02@
>functional_1/sequential/encoder_conv5/BiasAdd_1/ReadVariableOp�
/functional_1/sequential/encoder_conv5/BiasAdd_1BiasAdd7functional_1/sequential/encoder_conv5/Conv2D_1:output:0Ffunctional_1/sequential/encoder_conv5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P21
/functional_1/sequential/encoder_conv5/BiasAdd_1�
+functional_1/sequential/encoder_conv5/Elu_1Elu8functional_1/sequential/encoder_conv5/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������P2-
+functional_1/sequential/encoder_conv5/Elu_1�
/functional_1/sequential/encoder_pool5/MaxPool_1MaxPool9functional_1/sequential/encoder_conv5/Elu_1:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
21
/functional_1/sequential/encoder_pool5/MaxPool_1�
3functional_1/sequential/encoder_dropout5/Identity_1Identity8functional_1/sequential/encoder_pool5/MaxPool_1:output:0*
T0*/
_output_shapes
:���������P25
3functional_1/sequential/encoder_dropout5/Identity_1�
'functional_1/sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   2)
'functional_1/sequential/flatten/Const_1�
)functional_1/sequential/flatten/Reshape_1Reshape<functional_1/sequential/encoder_dropout5/Identity_1:output:00functional_1/sequential/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������
2+
)functional_1/sequential/flatten/Reshape_1�
5functional_1/sequential/dense/MatMul_1/ReadVariableOpReadVariableOp<functional_1_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype027
5functional_1/sequential/dense/MatMul_1/ReadVariableOp�
&functional_1/sequential/dense/MatMul_1MatMul2functional_1/sequential/flatten/Reshape_1:output:0=functional_1/sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&functional_1/sequential/dense/MatMul_1�
6functional_1/sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp=functional_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype028
6functional_1/sequential/dense/BiasAdd_1/ReadVariableOp�
'functional_1/sequential/dense/BiasAdd_1BiasAdd0functional_1/sequential/dense/MatMul_1:product:0>functional_1/sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'functional_1/sequential/dense/BiasAdd_1�
'functional_1/sequential/dense/Sigmoid_1Sigmoid0functional_1/sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2)
'functional_1/sequential/dense/Sigmoid_1�
functional_1/subtract/subSub)functional_1/sequential/dense/Sigmoid:y:0+functional_1/sequential/dense/Sigmoid_1:y:0*
T0*(
_output_shapes
:����������2
functional_1/subtract/sub�
 functional_1/tf_op_layer_Abs/AbsAbsfunctional_1/subtract/sub:z:0*
T0*
_cloned(*(
_output_shapes
:����������2"
 functional_1/tf_op_layer_Abs/Abs�
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOp�
functional_1/dense_1/MatMulMatMul$functional_1/tf_op_layer_Abs/Abs:y:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
functional_1/dense_1/MatMul�
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOp�
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
functional_1/dense_1/BiasAdd�
functional_1/dense_1/SigmoidSigmoid%functional_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
functional_1/dense_1/Sigmoidt
IdentityIdentity functional_1/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������:::::::::::::::Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
b
F__inference_random_flip_layer_call_and_return_conditional_losses_13988

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
G
+__inference_random_flip_layer_call_fn_16602

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_138902
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17113

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������P2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�
T
(__inference_subtract_layer_call_fn_16506
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_149142
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
d
+__inference_random_flip_layer_call_fn_16597

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_138812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_16967

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������   2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������   *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������   2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������   2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������   2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
|
'__inference_dense_1_layer_call_fn_16536

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_149472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_16511

inputs
identity[
AbsAbsinputs*
T0*
_cloned(*(
_output_shapes
:����������2
Abs\
IdentityIdentityAbs:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_random_flip_layer_call_and_return_conditional_losses_16634

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_16993

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������  02
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������  02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   :::W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�(
e
F__inference_random_flip_layer_call_and_return_conditional_losses_13984

inputs
identity��
)random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:�����������2+
)random_flip_left_right/control_dependency�
random_flip_left_right/ShapeShape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2
random_flip_left_right/Shape�
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*random_flip_left_right/strided_slice/stack�
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_1�
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2�
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$random_flip_left_right/strided_slice�
+random_flip_left_right/random_uniform/shapePack-random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape�
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min�
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)random_flip_left_right/random_uniform/max�
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype025
3random_flip_left_right/random_uniform/RandomUniform�
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:���������2+
)random_flip_left_right/random_uniform/Mul�
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1�
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2�
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3�
$random_flip_left_right/Reshape/shapePack-random_flip_left_right/strided_slice:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shape�
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2 
random_flip_left_right/Reshape�
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/Round�
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axis�
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:�����������2"
 random_flip_left_right/ReverseV2�
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:�����������2
random_flip_left_right/mul�
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_flip_left_right/sub/x�
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/sub�
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:�����������2
random_flip_left_right/mul_1�
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:�����������2
random_flip_left_right/add|
IdentityIdentityrandom_flip_left_right/add:z:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_13923

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_14531

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������P2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������P2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�
d
+__inference_random_flip_layer_call_fn_16639

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_139842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
z
%__inference_dense_layer_call_fn_17154

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_145692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������
::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_14468

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_16527

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_17129

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�
i
0__inference_encoder_dropout2_layer_call_fn_16977

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_143522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
L
0__inference_encoder_dropout5_layer_call_fn_17123

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_145312
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
F__inference_random_zoom_layer_call_and_return_conditional_losses_16751

inputs-
)stateful_uniform_statefuluniform_resource
identity�� stateful_uniform/StatefulUniform�"stateful_uniform_1/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1�
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
stateful_uniform/max�
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm�
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype02"
 stateful_uniform/StatefulUniform�
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub�
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform/mul�
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniformz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1�
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform_1/shapeu
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
stateful_uniform_1/max�
,stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2.
,stateful_uniform_1/StatefulUniform/algorithm�
"stateful_uniform_1/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource5stateful_uniform_1/StatefulUniform/algorithm:output:0!stateful_uniform_1/shape:output:0!^stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02$
"stateful_uniform_1/StatefulUniform�
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub�
stateful_uniform_1/mulMul+stateful_uniform_1/StatefulUniform:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1/mul�
stateful_uniform_1Addstateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2stateful_uniform_1:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shape�
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stack�
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1�
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2�
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zoom_matrix/strided_slicek
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub/yr
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/subs
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv/y�
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv�
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stack�
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1�
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2�
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_1o
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub_1/x�
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/sub_1�
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:���������2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub_2/yv
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/sub_2w
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv_1/y�
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1�
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stack�
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1�
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2�
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_2o
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub_3/x�
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/sub_3�
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:���������2
zoom_matrix/mul_1�
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stack�
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1�
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2�
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_3t
zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/mul/y�
zoom_matrix/zeros/mulMul"zoom_matrix/strided_slice:output:0 zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros/mulw
zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zoom_matrix/zeros/Less/y�
zoom_matrix/zeros/LessLesszoom_matrix/zeros/mul:z:0!zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros/Lessz
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/packed/1�
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros/packedw
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros/Const�
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/zerosx
zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/mul/y�
zoom_matrix/zeros_1/mulMul"zoom_matrix/strided_slice:output:0"zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_1/mul{
zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zoom_matrix/zeros_1/Less/y�
zoom_matrix/zeros_1/LessLesszoom_matrix/zeros_1/mul:z:0#zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_1/Less~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1�
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_1/packed{
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_1/Const�
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/zeros_1�
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stack�
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1�
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2�
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_4x
zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/mul/y�
zoom_matrix/zeros_2/mulMul"zoom_matrix/strided_slice:output:0"zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_2/mul{
zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zoom_matrix/zeros_2/Less/y�
zoom_matrix/zeros_2/LessLesszoom_matrix/zeros_2/mul:z:0#zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_2/Less~
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/packed/1�
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_2/packed{
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_2/Const�
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axis�
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape�
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack�
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1�
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice�
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputszoom_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2�
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform#^stateful_uniform_1/StatefulUniform*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform2H
"stateful_uniform_1/StatefulUniform"stateful_uniform_1/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
F__inference_random_flip_layer_call_and_return_conditional_losses_13890

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_14569

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������
:::P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_14439

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������0:::W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
G
+__inference_random_zoom_layer_call_fn_16767

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_141112
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��	
�
G__inference_functional_1_layer_call_and_return_conditional_losses_15871
inputs_0
inputs_1D
@sequential_random_zoom_stateful_uniform_statefuluniform_resourceK
Gsequential_random_translation_stateful_uniform_statefuluniform_resource;
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
identity��>sequential/random_translation/stateful_uniform/StatefulUniform�@sequential/random_translation/stateful_uniform_1/StatefulUniform�@sequential/random_translation/stateful_uniform_2/StatefulUniform�@sequential/random_translation/stateful_uniform_3/StatefulUniform�7sequential/random_zoom/stateful_uniform/StatefulUniform�9sequential/random_zoom/stateful_uniform_1/StatefulUniform�9sequential/random_zoom/stateful_uniform_2/StatefulUniform�9sequential/random_zoom/stateful_uniform_3/StatefulUniform�
@sequential/random_flip/random_flip_left_right/control_dependencyIdentityinputs_0*
T0*
_class
loc:@inputs/0*1
_output_shapes
:�����������2B
@sequential/random_flip/random_flip_left_right/control_dependency�
3sequential/random_flip/random_flip_left_right/ShapeShapeIsequential/random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:25
3sequential/random_flip/random_flip_left_right/Shape�
Asequential/random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/random_flip/random_flip_left_right/strided_slice/stack�
Csequential/random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential/random_flip/random_flip_left_right/strided_slice/stack_1�
Csequential/random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential/random_flip/random_flip_left_right/strided_slice/stack_2�
;sequential/random_flip/random_flip_left_right/strided_sliceStridedSlice<sequential/random_flip/random_flip_left_right/Shape:output:0Jsequential/random_flip/random_flip_left_right/strided_slice/stack:output:0Lsequential/random_flip/random_flip_left_right/strided_slice/stack_1:output:0Lsequential/random_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;sequential/random_flip/random_flip_left_right/strided_slice�
Bsequential/random_flip/random_flip_left_right/random_uniform/shapePackDsequential/random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2D
Bsequential/random_flip/random_flip_left_right/random_uniform/shape�
@sequential/random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@sequential/random_flip/random_flip_left_right/random_uniform/min�
@sequential/random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2B
@sequential/random_flip/random_flip_left_right/random_uniform/max�
Jsequential/random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniformKsequential/random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype02L
Jsequential/random_flip/random_flip_left_right/random_uniform/RandomUniform�
@sequential/random_flip/random_flip_left_right/random_uniform/MulMulSsequential/random_flip/random_flip_left_right/random_uniform/RandomUniform:output:0Isequential/random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:���������2B
@sequential/random_flip/random_flip_left_right/random_uniform/Mul�
=sequential/random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/random_flip/random_flip_left_right/Reshape/shape/1�
=sequential/random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/random_flip/random_flip_left_right/Reshape/shape/2�
=sequential/random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/random_flip/random_flip_left_right/Reshape/shape/3�
;sequential/random_flip/random_flip_left_right/Reshape/shapePackDsequential/random_flip/random_flip_left_right/strided_slice:output:0Fsequential/random_flip/random_flip_left_right/Reshape/shape/1:output:0Fsequential/random_flip/random_flip_left_right/Reshape/shape/2:output:0Fsequential/random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2=
;sequential/random_flip/random_flip_left_right/Reshape/shape�
5sequential/random_flip/random_flip_left_right/ReshapeReshapeDsequential/random_flip/random_flip_left_right/random_uniform/Mul:z:0Dsequential/random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:���������27
5sequential/random_flip/random_flip_left_right/Reshape�
3sequential/random_flip/random_flip_left_right/RoundRound>sequential/random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������25
3sequential/random_flip/random_flip_left_right/Round�
<sequential/random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential/random_flip/random_flip_left_right/ReverseV2/axis�
7sequential/random_flip/random_flip_left_right/ReverseV2	ReverseV2Isequential/random_flip/random_flip_left_right/control_dependency:output:0Esequential/random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:�����������29
7sequential/random_flip/random_flip_left_right/ReverseV2�
1sequential/random_flip/random_flip_left_right/mulMul7sequential/random_flip/random_flip_left_right/Round:y:0@sequential/random_flip/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:�����������23
1sequential/random_flip/random_flip_left_right/mul�
3sequential/random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?25
3sequential/random_flip/random_flip_left_right/sub/x�
1sequential/random_flip/random_flip_left_right/subSub<sequential/random_flip/random_flip_left_right/sub/x:output:07sequential/random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:���������23
1sequential/random_flip/random_flip_left_right/sub�
3sequential/random_flip/random_flip_left_right/mul_1Mul5sequential/random_flip/random_flip_left_right/sub:z:0Isequential/random_flip/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:�����������25
3sequential/random_flip/random_flip_left_right/mul_1�
1sequential/random_flip/random_flip_left_right/addAddV25sequential/random_flip/random_flip_left_right/mul:z:07sequential/random_flip/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:�����������23
1sequential/random_flip/random_flip_left_right/add�
sequential/random_zoom/ShapeShape5sequential/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
sequential/random_zoom/Shape�
*sequential/random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential/random_zoom/strided_slice/stack�
,sequential/random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_zoom/strided_slice/stack_1�
,sequential/random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_zoom/strided_slice/stack_2�
$sequential/random_zoom/strided_sliceStridedSlice%sequential/random_zoom/Shape:output:03sequential/random_zoom/strided_slice/stack:output:05sequential/random_zoom/strided_slice/stack_1:output:05sequential/random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential/random_zoom/strided_slice�
,sequential/random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_zoom/strided_slice_1/stack�
.sequential/random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_1/stack_1�
.sequential/random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_1/stack_2�
&sequential/random_zoom/strided_slice_1StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_1/stack:output:07sequential/random_zoom/strided_slice_1/stack_1:output:07sequential/random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/random_zoom/strided_slice_1�
sequential/random_zoom/CastCast/sequential/random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
sequential/random_zoom/Cast�
,sequential/random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_zoom/strided_slice_2/stack�
.sequential/random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_2/stack_1�
.sequential/random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_2/stack_2�
&sequential/random_zoom/strided_slice_2StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_2/stack:output:07sequential/random_zoom/strided_slice_2/stack_1:output:07sequential/random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/random_zoom/strided_slice_2�
sequential/random_zoom/Cast_1Cast/sequential/random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
sequential/random_zoom/Cast_1�
/sequential/random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential/random_zoom/stateful_uniform/shape/1�
-sequential/random_zoom/stateful_uniform/shapePack-sequential/random_zoom/strided_slice:output:08sequential/random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-sequential/random_zoom/stateful_uniform/shape�
+sequential/random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2-
+sequential/random_zoom/stateful_uniform/min�
+sequential/random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2-
+sequential/random_zoom/stateful_uniform/max�
Asequential/random_zoom/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2C
Asequential/random_zoom/stateful_uniform/StatefulUniform/algorithm�
7sequential/random_zoom/stateful_uniform/StatefulUniformStatefulUniform@sequential_random_zoom_stateful_uniform_statefuluniform_resourceJsequential/random_zoom/stateful_uniform/StatefulUniform/algorithm:output:06sequential/random_zoom/stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype029
7sequential/random_zoom/stateful_uniform/StatefulUniform�
+sequential/random_zoom/stateful_uniform/subSub4sequential/random_zoom/stateful_uniform/max:output:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2-
+sequential/random_zoom/stateful_uniform/sub�
+sequential/random_zoom/stateful_uniform/mulMul@sequential/random_zoom/stateful_uniform/StatefulUniform:output:0/sequential/random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2-
+sequential/random_zoom/stateful_uniform/mul�
'sequential/random_zoom/stateful_uniformAdd/sequential/random_zoom/stateful_uniform/mul:z:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2)
'sequential/random_zoom/stateful_uniform�
1sequential/random_zoom/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential/random_zoom/stateful_uniform_1/shape/1�
/sequential/random_zoom/stateful_uniform_1/shapePack-sequential/random_zoom/strided_slice:output:0:sequential/random_zoom/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:21
/sequential/random_zoom/stateful_uniform_1/shape�
-sequential/random_zoom/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2/
-sequential/random_zoom/stateful_uniform_1/min�
-sequential/random_zoom/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2/
-sequential/random_zoom/stateful_uniform_1/max�
Csequential/random_zoom/stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
Csequential/random_zoom/stateful_uniform_1/StatefulUniform/algorithm�
9sequential/random_zoom/stateful_uniform_1/StatefulUniformStatefulUniform@sequential_random_zoom_stateful_uniform_statefuluniform_resourceLsequential/random_zoom/stateful_uniform_1/StatefulUniform/algorithm:output:08sequential/random_zoom/stateful_uniform_1/shape:output:08^sequential/random_zoom/stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02;
9sequential/random_zoom/stateful_uniform_1/StatefulUniform�
-sequential/random_zoom/stateful_uniform_1/subSub6sequential/random_zoom/stateful_uniform_1/max:output:06sequential/random_zoom/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2/
-sequential/random_zoom/stateful_uniform_1/sub�
-sequential/random_zoom/stateful_uniform_1/mulMulBsequential/random_zoom/stateful_uniform_1/StatefulUniform:output:01sequential/random_zoom/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2/
-sequential/random_zoom/stateful_uniform_1/mul�
)sequential/random_zoom/stateful_uniform_1Add1sequential/random_zoom/stateful_uniform_1/mul:z:06sequential/random_zoom/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2+
)sequential/random_zoom/stateful_uniform_1�
"sequential/random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/random_zoom/concat/axis�
sequential/random_zoom/concatConcatV2-sequential/random_zoom/stateful_uniform_1:z:0+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
sequential/random_zoom/concat�
(sequential/random_zoom/zoom_matrix/ShapeShape&sequential/random_zoom/concat:output:0*
T0*
_output_shapes
:2*
(sequential/random_zoom/zoom_matrix/Shape�
6sequential/random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential/random_zoom/zoom_matrix/strided_slice/stack�
8sequential/random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential/random_zoom/zoom_matrix/strided_slice/stack_1�
8sequential/random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential/random_zoom/zoom_matrix/strided_slice/stack_2�
0sequential/random_zoom/zoom_matrix/strided_sliceStridedSlice1sequential/random_zoom/zoom_matrix/Shape:output:0?sequential/random_zoom/zoom_matrix/strided_slice/stack:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_1:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential/random_zoom/zoom_matrix/strided_slice�
(sequential/random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2*
(sequential/random_zoom/zoom_matrix/sub/y�
&sequential/random_zoom/zoom_matrix/subSub!sequential/random_zoom/Cast_1:y:01sequential/random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2(
&sequential/random_zoom/zoom_matrix/sub�
,sequential/random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,sequential/random_zoom/zoom_matrix/truediv/y�
*sequential/random_zoom/zoom_matrix/truedivRealDiv*sequential/random_zoom/zoom_matrix/sub:z:05sequential/random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2,
*sequential/random_zoom/zoom_matrix/truediv�
8sequential/random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2:
8sequential/random_zoom/zoom_matrix/strided_slice_1/stack�
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2<
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1�
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2<
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2�
2sequential/random_zoom/zoom_matrix/strided_slice_1StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_1/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask24
2sequential/random_zoom/zoom_matrix/strided_slice_1�
*sequential/random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2,
*sequential/random_zoom/zoom_matrix/sub_1/x�
(sequential/random_zoom/zoom_matrix/sub_1Sub3sequential/random_zoom/zoom_matrix/sub_1/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2*
(sequential/random_zoom/zoom_matrix/sub_1�
&sequential/random_zoom/zoom_matrix/mulMul.sequential/random_zoom/zoom_matrix/truediv:z:0,sequential/random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:���������2(
&sequential/random_zoom/zoom_matrix/mul�
*sequential/random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2,
*sequential/random_zoom/zoom_matrix/sub_2/y�
(sequential/random_zoom/zoom_matrix/sub_2Subsequential/random_zoom/Cast:y:03sequential/random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2*
(sequential/random_zoom/zoom_matrix/sub_2�
.sequential/random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.sequential/random_zoom/zoom_matrix/truediv_1/y�
,sequential/random_zoom/zoom_matrix/truediv_1RealDiv,sequential/random_zoom/zoom_matrix/sub_2:z:07sequential/random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2.
,sequential/random_zoom/zoom_matrix/truediv_1�
8sequential/random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2:
8sequential/random_zoom/zoom_matrix/strided_slice_2/stack�
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2<
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1�
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2<
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2�
2sequential/random_zoom/zoom_matrix/strided_slice_2StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_2/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask24
2sequential/random_zoom/zoom_matrix/strided_slice_2�
*sequential/random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2,
*sequential/random_zoom/zoom_matrix/sub_3/x�
(sequential/random_zoom/zoom_matrix/sub_3Sub3sequential/random_zoom/zoom_matrix/sub_3/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2*
(sequential/random_zoom/zoom_matrix/sub_3�
(sequential/random_zoom/zoom_matrix/mul_1Mul0sequential/random_zoom/zoom_matrix/truediv_1:z:0,sequential/random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:���������2*
(sequential/random_zoom/zoom_matrix/mul_1�
8sequential/random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2:
8sequential/random_zoom/zoom_matrix/strided_slice_3/stack�
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2<
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1�
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2<
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2�
2sequential/random_zoom/zoom_matrix/strided_slice_3StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_3/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask24
2sequential/random_zoom/zoom_matrix/strided_slice_3�
.sequential/random_zoom/zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential/random_zoom/zoom_matrix/zeros/mul/y�
,sequential/random_zoom/zoom_matrix/zeros/mulMul9sequential/random_zoom/zoom_matrix/strided_slice:output:07sequential/random_zoom/zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2.
,sequential/random_zoom/zoom_matrix/zeros/mul�
/sequential/random_zoom/zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�21
/sequential/random_zoom/zoom_matrix/zeros/Less/y�
-sequential/random_zoom/zoom_matrix/zeros/LessLess0sequential/random_zoom/zoom_matrix/zeros/mul:z:08sequential/random_zoom/zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2/
-sequential/random_zoom/zoom_matrix/zeros/Less�
1sequential/random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential/random_zoom/zoom_matrix/zeros/packed/1�
/sequential/random_zoom/zoom_matrix/zeros/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0:sequential/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:21
/sequential/random_zoom/zoom_matrix/zeros/packed�
.sequential/random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.sequential/random_zoom/zoom_matrix/zeros/Const�
(sequential/random_zoom/zoom_matrix/zerosFill8sequential/random_zoom/zoom_matrix/zeros/packed:output:07sequential/random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2*
(sequential/random_zoom/zoom_matrix/zeros�
0sequential/random_zoom/zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/random_zoom/zoom_matrix/zeros_1/mul/y�
.sequential/random_zoom/zoom_matrix/zeros_1/mulMul9sequential/random_zoom/zoom_matrix/strided_slice:output:09sequential/random_zoom/zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 20
.sequential/random_zoom/zoom_matrix/zeros_1/mul�
1sequential/random_zoom/zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�23
1sequential/random_zoom/zoom_matrix/zeros_1/Less/y�
/sequential/random_zoom/zoom_matrix/zeros_1/LessLess2sequential/random_zoom/zoom_matrix/zeros_1/mul:z:0:sequential/random_zoom/zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 21
/sequential/random_zoom/zoom_matrix/zeros_1/Less�
3sequential/random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential/random_zoom/zoom_matrix/zeros_1/packed/1�
1sequential/random_zoom/zoom_matrix/zeros_1/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:23
1sequential/random_zoom/zoom_matrix/zeros_1/packed�
0sequential/random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0sequential/random_zoom/zoom_matrix/zeros_1/Const�
*sequential/random_zoom/zoom_matrix/zeros_1Fill:sequential/random_zoom/zoom_matrix/zeros_1/packed:output:09sequential/random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2,
*sequential/random_zoom/zoom_matrix/zeros_1�
8sequential/random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2:
8sequential/random_zoom/zoom_matrix/strided_slice_4/stack�
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2<
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1�
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2<
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2�
2sequential/random_zoom/zoom_matrix/strided_slice_4StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_4/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask24
2sequential/random_zoom/zoom_matrix/strided_slice_4�
0sequential/random_zoom/zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/random_zoom/zoom_matrix/zeros_2/mul/y�
.sequential/random_zoom/zoom_matrix/zeros_2/mulMul9sequential/random_zoom/zoom_matrix/strided_slice:output:09sequential/random_zoom/zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 20
.sequential/random_zoom/zoom_matrix/zeros_2/mul�
1sequential/random_zoom/zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�23
1sequential/random_zoom/zoom_matrix/zeros_2/Less/y�
/sequential/random_zoom/zoom_matrix/zeros_2/LessLess2sequential/random_zoom/zoom_matrix/zeros_2/mul:z:0:sequential/random_zoom/zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 21
/sequential/random_zoom/zoom_matrix/zeros_2/Less�
3sequential/random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential/random_zoom/zoom_matrix/zeros_2/packed/1�
1sequential/random_zoom/zoom_matrix/zeros_2/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:23
1sequential/random_zoom/zoom_matrix/zeros_2/packed�
0sequential/random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0sequential/random_zoom/zoom_matrix/zeros_2/Const�
*sequential/random_zoom/zoom_matrix/zeros_2Fill:sequential/random_zoom/zoom_matrix/zeros_2/packed:output:09sequential/random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2,
*sequential/random_zoom/zoom_matrix/zeros_2�
.sequential/random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential/random_zoom/zoom_matrix/concat/axis�
)sequential/random_zoom/zoom_matrix/concatConcatV2;sequential/random_zoom/zoom_matrix/strided_slice_3:output:01sequential/random_zoom/zoom_matrix/zeros:output:0*sequential/random_zoom/zoom_matrix/mul:z:03sequential/random_zoom/zoom_matrix/zeros_1:output:0;sequential/random_zoom/zoom_matrix/strided_slice_4:output:0,sequential/random_zoom/zoom_matrix/mul_1:z:03sequential/random_zoom/zoom_matrix/zeros_2:output:07sequential/random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2+
)sequential/random_zoom/zoom_matrix/concat�
&sequential/random_zoom/transform/ShapeShape5sequential/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2(
&sequential/random_zoom/transform/Shape�
4sequential/random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential/random_zoom/transform/strided_slice/stack�
6sequential/random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/random_zoom/transform/strided_slice/stack_1�
6sequential/random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/random_zoom/transform/strided_slice/stack_2�
.sequential/random_zoom/transform/strided_sliceStridedSlice/sequential/random_zoom/transform/Shape:output:0=sequential/random_zoom/transform/strided_slice/stack:output:0?sequential/random_zoom/transform/strided_slice/stack_1:output:0?sequential/random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:20
.sequential/random_zoom/transform/strided_slice�
;sequential/random_zoom/transform/ImageProjectiveTransformV2ImageProjectiveTransformV25sequential/random_flip/random_flip_left_right/add:z:02sequential/random_zoom/zoom_matrix/concat:output:07sequential/random_zoom/transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2=
;sequential/random_zoom/transform/ImageProjectiveTransformV2�
#sequential/random_translation/ShapeShapePsequential/random_zoom/transform/ImageProjectiveTransformV2:transformed_images:0*
T0*
_output_shapes
:2%
#sequential/random_translation/Shape�
1sequential/random_translation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/random_translation/strided_slice/stack�
3sequential/random_translation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/random_translation/strided_slice/stack_1�
3sequential/random_translation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/random_translation/strided_slice/stack_2�
+sequential/random_translation/strided_sliceStridedSlice,sequential/random_translation/Shape:output:0:sequential/random_translation/strided_slice/stack:output:0<sequential/random_translation/strided_slice/stack_1:output:0<sequential/random_translation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/random_translation/strided_slice�
3sequential/random_translation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/random_translation/strided_slice_1/stack�
5sequential/random_translation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_1/stack_1�
5sequential/random_translation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_1/stack_2�
-sequential/random_translation/strided_slice_1StridedSlice,sequential/random_translation/Shape:output:0<sequential/random_translation/strided_slice_1/stack:output:0>sequential/random_translation/strided_slice_1/stack_1:output:0>sequential/random_translation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/random_translation/strided_slice_1�
"sequential/random_translation/CastCast6sequential/random_translation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"sequential/random_translation/Cast�
3sequential/random_translation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/random_translation/strided_slice_2/stack�
5sequential/random_translation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_2/stack_1�
5sequential/random_translation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_2/stack_2�
-sequential/random_translation/strided_slice_2StridedSlice,sequential/random_translation/Shape:output:0<sequential/random_translation/strided_slice_2/stack:output:0>sequential/random_translation/strided_slice_2/stack_1:output:0>sequential/random_translation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/random_translation/strided_slice_2�
$sequential/random_translation/Cast_1Cast6sequential/random_translation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$sequential/random_translation/Cast_1�
6sequential/random_translation/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :28
6sequential/random_translation/stateful_uniform/shape/1�
4sequential/random_translation/stateful_uniform/shapePack4sequential/random_translation/strided_slice:output:0?sequential/random_translation/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:26
4sequential/random_translation/stateful_uniform/shape�
2sequential/random_translation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�24
2sequential/random_translation/stateful_uniform/min�
2sequential/random_translation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2sequential/random_translation/stateful_uniform/max�
Hsequential/random_translation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2J
Hsequential/random_translation/stateful_uniform/StatefulUniform/algorithm�
>sequential/random_translation/stateful_uniform/StatefulUniformStatefulUniformGsequential_random_translation_stateful_uniform_statefuluniform_resourceQsequential/random_translation/stateful_uniform/StatefulUniform/algorithm:output:0=sequential/random_translation/stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype02@
>sequential/random_translation/stateful_uniform/StatefulUniform�
2sequential/random_translation/stateful_uniform/subSub;sequential/random_translation/stateful_uniform/max:output:0;sequential/random_translation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 24
2sequential/random_translation/stateful_uniform/sub�
2sequential/random_translation/stateful_uniform/mulMulGsequential/random_translation/stateful_uniform/StatefulUniform:output:06sequential/random_translation/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������24
2sequential/random_translation/stateful_uniform/mul�
.sequential/random_translation/stateful_uniformAdd6sequential/random_translation/stateful_uniform/mul:z:0;sequential/random_translation/stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������20
.sequential/random_translation/stateful_uniform�
!sequential/random_translation/mulMul2sequential/random_translation/stateful_uniform:z:0&sequential/random_translation/Cast:y:0*
T0*'
_output_shapes
:���������2#
!sequential/random_translation/mul�
8sequential/random_translation/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8sequential/random_translation/stateful_uniform_1/shape/1�
6sequential/random_translation/stateful_uniform_1/shapePack4sequential/random_translation/strided_slice:output:0Asequential/random_translation/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:28
6sequential/random_translation/stateful_uniform_1/shape�
4sequential/random_translation/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�26
4sequential/random_translation/stateful_uniform_1/min�
4sequential/random_translation/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/random_translation/stateful_uniform_1/max�
Jsequential/random_translation/stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2L
Jsequential/random_translation/stateful_uniform_1/StatefulUniform/algorithm�
@sequential/random_translation/stateful_uniform_1/StatefulUniformStatefulUniformGsequential_random_translation_stateful_uniform_statefuluniform_resourceSsequential/random_translation/stateful_uniform_1/StatefulUniform/algorithm:output:0?sequential/random_translation/stateful_uniform_1/shape:output:0?^sequential/random_translation/stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02B
@sequential/random_translation/stateful_uniform_1/StatefulUniform�
4sequential/random_translation/stateful_uniform_1/subSub=sequential/random_translation/stateful_uniform_1/max:output:0=sequential/random_translation/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 26
4sequential/random_translation/stateful_uniform_1/sub�
4sequential/random_translation/stateful_uniform_1/mulMulIsequential/random_translation/stateful_uniform_1/StatefulUniform:output:08sequential/random_translation/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������26
4sequential/random_translation/stateful_uniform_1/mul�
0sequential/random_translation/stateful_uniform_1Add8sequential/random_translation/stateful_uniform_1/mul:z:0=sequential/random_translation/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������22
0sequential/random_translation/stateful_uniform_1�
#sequential/random_translation/mul_1Mul4sequential/random_translation/stateful_uniform_1:z:0(sequential/random_translation/Cast_1:y:0*
T0*'
_output_shapes
:���������2%
#sequential/random_translation/mul_1�
)sequential/random_translation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/random_translation/concat/axis�
$sequential/random_translation/concatConcatV2'sequential/random_translation/mul_1:z:0%sequential/random_translation/mul:z:02sequential/random_translation/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2&
$sequential/random_translation/concat�
6sequential/random_translation/translation_matrix/ShapeShape-sequential/random_translation/concat:output:0*
T0*
_output_shapes
:28
6sequential/random_translation/translation_matrix/Shape�
Dsequential/random_translation/translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential/random_translation/translation_matrix/strided_slice/stack�
Fsequential/random_translation/translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential/random_translation/translation_matrix/strided_slice/stack_1�
Fsequential/random_translation/translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential/random_translation/translation_matrix/strided_slice/stack_2�
>sequential/random_translation/translation_matrix/strided_sliceStridedSlice?sequential/random_translation/translation_matrix/Shape:output:0Msequential/random_translation/translation_matrix/strided_slice/stack:output:0Osequential/random_translation/translation_matrix/strided_slice/stack_1:output:0Osequential/random_translation/translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential/random_translation/translation_matrix/strided_slice�
;sequential/random_translation/translation_matrix/ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential/random_translation/translation_matrix/ones/mul/y�
9sequential/random_translation/translation_matrix/ones/mulMulGsequential/random_translation/translation_matrix/strided_slice:output:0Dsequential/random_translation/translation_matrix/ones/mul/y:output:0*
T0*
_output_shapes
: 2;
9sequential/random_translation/translation_matrix/ones/mul�
<sequential/random_translation/translation_matrix/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2>
<sequential/random_translation/translation_matrix/ones/Less/y�
:sequential/random_translation/translation_matrix/ones/LessLess=sequential/random_translation/translation_matrix/ones/mul:z:0Esequential/random_translation/translation_matrix/ones/Less/y:output:0*
T0*
_output_shapes
: 2<
:sequential/random_translation/translation_matrix/ones/Less�
>sequential/random_translation/translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/random_translation/translation_matrix/ones/packed/1�
<sequential/random_translation/translation_matrix/ones/packedPackGsequential/random_translation/translation_matrix/strided_slice:output:0Gsequential/random_translation/translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential/random_translation/translation_matrix/ones/packed�
;sequential/random_translation/translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2=
;sequential/random_translation/translation_matrix/ones/Const�
5sequential/random_translation/translation_matrix/onesFillEsequential/random_translation/translation_matrix/ones/packed:output:0Dsequential/random_translation/translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:���������27
5sequential/random_translation/translation_matrix/ones�
<sequential/random_translation/translation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential/random_translation/translation_matrix/zeros/mul/y�
:sequential/random_translation/translation_matrix/zeros/mulMulGsequential/random_translation/translation_matrix/strided_slice:output:0Esequential/random_translation/translation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2<
:sequential/random_translation/translation_matrix/zeros/mul�
=sequential/random_translation/translation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2?
=sequential/random_translation/translation_matrix/zeros/Less/y�
;sequential/random_translation/translation_matrix/zeros/LessLess>sequential/random_translation/translation_matrix/zeros/mul:z:0Fsequential/random_translation/translation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2=
;sequential/random_translation/translation_matrix/zeros/Less�
?sequential/random_translation/translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?sequential/random_translation/translation_matrix/zeros/packed/1�
=sequential/random_translation/translation_matrix/zeros/packedPackGsequential/random_translation/translation_matrix/strided_slice:output:0Hsequential/random_translation/translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential/random_translation/translation_matrix/zeros/packed�
<sequential/random_translation/translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<sequential/random_translation/translation_matrix/zeros/Const�
6sequential/random_translation/translation_matrix/zerosFillFsequential/random_translation/translation_matrix/zeros/packed:output:0Esequential/random_translation/translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������28
6sequential/random_translation/translation_matrix/zeros�
Fsequential/random_translation/translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2H
Fsequential/random_translation/translation_matrix/strided_slice_1/stack�
Hsequential/random_translation/translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2J
Hsequential/random_translation/translation_matrix/strided_slice_1/stack_1�
Hsequential/random_translation/translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2J
Hsequential/random_translation/translation_matrix/strided_slice_1/stack_2�
@sequential/random_translation/translation_matrix/strided_slice_1StridedSlice-sequential/random_translation/concat:output:0Osequential/random_translation/translation_matrix/strided_slice_1/stack:output:0Qsequential/random_translation/translation_matrix/strided_slice_1/stack_1:output:0Qsequential/random_translation/translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2B
@sequential/random_translation/translation_matrix/strided_slice_1�
4sequential/random_translation/translation_matrix/NegNegIsequential/random_translation/translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������26
4sequential/random_translation/translation_matrix/Neg�
>sequential/random_translation/translation_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/random_translation/translation_matrix/zeros_1/mul/y�
<sequential/random_translation/translation_matrix/zeros_1/mulMulGsequential/random_translation/translation_matrix/strided_slice:output:0Gsequential/random_translation/translation_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential/random_translation/translation_matrix/zeros_1/mul�
?sequential/random_translation/translation_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2A
?sequential/random_translation/translation_matrix/zeros_1/Less/y�
=sequential/random_translation/translation_matrix/zeros_1/LessLess@sequential/random_translation/translation_matrix/zeros_1/mul:z:0Hsequential/random_translation/translation_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential/random_translation/translation_matrix/zeros_1/Less�
Asequential/random_translation/translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential/random_translation/translation_matrix/zeros_1/packed/1�
?sequential/random_translation/translation_matrix/zeros_1/packedPackGsequential/random_translation/translation_matrix/strided_slice:output:0Jsequential/random_translation/translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential/random_translation/translation_matrix/zeros_1/packed�
>sequential/random_translation/translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>sequential/random_translation/translation_matrix/zeros_1/Const�
8sequential/random_translation/translation_matrix/zeros_1FillHsequential/random_translation/translation_matrix/zeros_1/packed:output:0Gsequential/random_translation/translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2:
8sequential/random_translation/translation_matrix/zeros_1�
=sequential/random_translation/translation_matrix/ones_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/random_translation/translation_matrix/ones_1/mul/y�
;sequential/random_translation/translation_matrix/ones_1/mulMulGsequential/random_translation/translation_matrix/strided_slice:output:0Fsequential/random_translation/translation_matrix/ones_1/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential/random_translation/translation_matrix/ones_1/mul�
>sequential/random_translation/translation_matrix/ones_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2@
>sequential/random_translation/translation_matrix/ones_1/Less/y�
<sequential/random_translation/translation_matrix/ones_1/LessLess?sequential/random_translation/translation_matrix/ones_1/mul:z:0Gsequential/random_translation/translation_matrix/ones_1/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential/random_translation/translation_matrix/ones_1/Less�
@sequential/random_translation/translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2B
@sequential/random_translation/translation_matrix/ones_1/packed/1�
>sequential/random_translation/translation_matrix/ones_1/packedPackGsequential/random_translation/translation_matrix/strided_slice:output:0Isequential/random_translation/translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential/random_translation/translation_matrix/ones_1/packed�
=sequential/random_translation/translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2?
=sequential/random_translation/translation_matrix/ones_1/Const�
7sequential/random_translation/translation_matrix/ones_1FillGsequential/random_translation/translation_matrix/ones_1/packed:output:0Fsequential/random_translation/translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:���������29
7sequential/random_translation/translation_matrix/ones_1�
Fsequential/random_translation/translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2H
Fsequential/random_translation/translation_matrix/strided_slice_2/stack�
Hsequential/random_translation/translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2J
Hsequential/random_translation/translation_matrix/strided_slice_2/stack_1�
Hsequential/random_translation/translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2J
Hsequential/random_translation/translation_matrix/strided_slice_2/stack_2�
@sequential/random_translation/translation_matrix/strided_slice_2StridedSlice-sequential/random_translation/concat:output:0Osequential/random_translation/translation_matrix/strided_slice_2/stack:output:0Qsequential/random_translation/translation_matrix/strided_slice_2/stack_1:output:0Qsequential/random_translation/translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2B
@sequential/random_translation/translation_matrix/strided_slice_2�
6sequential/random_translation/translation_matrix/Neg_1NegIsequential/random_translation/translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������28
6sequential/random_translation/translation_matrix/Neg_1�
>sequential/random_translation/translation_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/random_translation/translation_matrix/zeros_2/mul/y�
<sequential/random_translation/translation_matrix/zeros_2/mulMulGsequential/random_translation/translation_matrix/strided_slice:output:0Gsequential/random_translation/translation_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential/random_translation/translation_matrix/zeros_2/mul�
?sequential/random_translation/translation_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2A
?sequential/random_translation/translation_matrix/zeros_2/Less/y�
=sequential/random_translation/translation_matrix/zeros_2/LessLess@sequential/random_translation/translation_matrix/zeros_2/mul:z:0Hsequential/random_translation/translation_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential/random_translation/translation_matrix/zeros_2/Less�
Asequential/random_translation/translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential/random_translation/translation_matrix/zeros_2/packed/1�
?sequential/random_translation/translation_matrix/zeros_2/packedPackGsequential/random_translation/translation_matrix/strided_slice:output:0Jsequential/random_translation/translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential/random_translation/translation_matrix/zeros_2/packed�
>sequential/random_translation/translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>sequential/random_translation/translation_matrix/zeros_2/Const�
8sequential/random_translation/translation_matrix/zeros_2FillHsequential/random_translation/translation_matrix/zeros_2/packed:output:0Gsequential/random_translation/translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2:
8sequential/random_translation/translation_matrix/zeros_2�
<sequential/random_translation/translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential/random_translation/translation_matrix/concat/axis�
7sequential/random_translation/translation_matrix/concatConcatV2>sequential/random_translation/translation_matrix/ones:output:0?sequential/random_translation/translation_matrix/zeros:output:08sequential/random_translation/translation_matrix/Neg:y:0Asequential/random_translation/translation_matrix/zeros_1:output:0@sequential/random_translation/translation_matrix/ones_1:output:0:sequential/random_translation/translation_matrix/Neg_1:y:0Asequential/random_translation/translation_matrix/zeros_2:output:0Esequential/random_translation/translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������29
7sequential/random_translation/translation_matrix/concat�
-sequential/random_translation/transform/ShapeShapePsequential/random_zoom/transform/ImageProjectiveTransformV2:transformed_images:0*
T0*
_output_shapes
:2/
-sequential/random_translation/transform/Shape�
;sequential/random_translation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;sequential/random_translation/transform/strided_slice/stack�
=sequential/random_translation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/random_translation/transform/strided_slice/stack_1�
=sequential/random_translation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/random_translation/transform/strided_slice/stack_2�
5sequential/random_translation/transform/strided_sliceStridedSlice6sequential/random_translation/transform/Shape:output:0Dsequential/random_translation/transform/strided_slice/stack:output:0Fsequential/random_translation/transform/strided_slice/stack_1:output:0Fsequential/random_translation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:27
5sequential/random_translation/transform/strided_slice�
Bsequential/random_translation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2Psequential/random_zoom/transform/ImageProjectiveTransformV2:transformed_images:0@sequential/random_translation/translation_matrix/concat:output:0>sequential/random_translation/transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2D
Bsequential/random_translation/transform/ImageProjectiveTransformV2�
.sequential/encoder_conv1/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.sequential/encoder_conv1/Conv2D/ReadVariableOp�
sequential/encoder_conv1/Conv2DConv2DWsequential/random_translation/transform/ImageProjectiveTransformV2:transformed_images:06sequential/encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2!
sequential/encoder_conv1/Conv2D�
/sequential/encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/encoder_conv1/BiasAdd/ReadVariableOp�
 sequential/encoder_conv1/BiasAddBiasAdd(sequential/encoder_conv1/Conv2D:output:07sequential/encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2"
 sequential/encoder_conv1/BiasAdd�
sequential/encoder_conv1/EluElu)sequential/encoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
sequential/encoder_conv1/Elu�
 sequential/encoder_pool1/MaxPoolMaxPool*sequential/encoder_conv1/Elu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool1/MaxPool�
)sequential/encoder_dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)sequential/encoder_dropout1/dropout/Const�
'sequential/encoder_dropout1/dropout/MulMul)sequential/encoder_pool1/MaxPool:output:02sequential/encoder_dropout1/dropout/Const:output:0*
T0*/
_output_shapes
:���������@@2)
'sequential/encoder_dropout1/dropout/Mul�
)sequential/encoder_dropout1/dropout/ShapeShape)sequential/encoder_pool1/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout1/dropout/Shape�
@sequential/encoder_dropout1/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout1/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype02B
@sequential/encoder_dropout1/dropout/random_uniform/RandomUniform�
2sequential/encoder_dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2sequential/encoder_dropout1/dropout/GreaterEqual/y�
0sequential/encoder_dropout1/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout1/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@@22
0sequential/encoder_dropout1/dropout/GreaterEqual�
(sequential/encoder_dropout1/dropout/CastCast4sequential/encoder_dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2*
(sequential/encoder_dropout1/dropout/Cast�
)sequential/encoder_dropout1/dropout/Mul_1Mul+sequential/encoder_dropout1/dropout/Mul:z:0,sequential/encoder_dropout1/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2+
)sequential/encoder_dropout1/dropout/Mul_1�
.sequential/encoder_conv2/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential/encoder_conv2/Conv2D/ReadVariableOp�
sequential/encoder_conv2/Conv2DConv2D-sequential/encoder_dropout1/dropout/Mul_1:z:06sequential/encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2!
sequential/encoder_conv2/Conv2D�
/sequential/encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/encoder_conv2/BiasAdd/ReadVariableOp�
 sequential/encoder_conv2/BiasAddBiasAdd(sequential/encoder_conv2/Conv2D:output:07sequential/encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2"
 sequential/encoder_conv2/BiasAdd�
sequential/encoder_conv2/EluElu)sequential/encoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
sequential/encoder_conv2/Elu�
 sequential/encoder_pool2/MaxPoolMaxPool*sequential/encoder_conv2/Elu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool2/MaxPool�
)sequential/encoder_dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)sequential/encoder_dropout2/dropout/Const�
'sequential/encoder_dropout2/dropout/MulMul)sequential/encoder_pool2/MaxPool:output:02sequential/encoder_dropout2/dropout/Const:output:0*
T0*/
_output_shapes
:���������   2)
'sequential/encoder_dropout2/dropout/Mul�
)sequential/encoder_dropout2/dropout/ShapeShape)sequential/encoder_pool2/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout2/dropout/Shape�
@sequential/encoder_dropout2/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout2/dropout/Shape:output:0*
T0*/
_output_shapes
:���������   *
dtype02B
@sequential/encoder_dropout2/dropout/random_uniform/RandomUniform�
2sequential/encoder_dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2sequential/encoder_dropout2/dropout/GreaterEqual/y�
0sequential/encoder_dropout2/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout2/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������   22
0sequential/encoder_dropout2/dropout/GreaterEqual�
(sequential/encoder_dropout2/dropout/CastCast4sequential/encoder_dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������   2*
(sequential/encoder_dropout2/dropout/Cast�
)sequential/encoder_dropout2/dropout/Mul_1Mul+sequential/encoder_dropout2/dropout/Mul:z:0,sequential/encoder_dropout2/dropout/Cast:y:0*
T0*/
_output_shapes
:���������   2+
)sequential/encoder_dropout2/dropout/Mul_1�
.sequential/encoder_conv3/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype020
.sequential/encoder_conv3/Conv2D/ReadVariableOp�
sequential/encoder_conv3/Conv2DConv2D-sequential/encoder_dropout2/dropout/Mul_1:z:06sequential/encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2!
sequential/encoder_conv3/Conv2D�
/sequential/encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/encoder_conv3/BiasAdd/ReadVariableOp�
 sequential/encoder_conv3/BiasAddBiasAdd(sequential/encoder_conv3/Conv2D:output:07sequential/encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02"
 sequential/encoder_conv3/BiasAdd�
sequential/encoder_conv3/EluElu)sequential/encoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:���������  02
sequential/encoder_conv3/Elu�
 sequential/encoder_pool3/MaxPoolMaxPool*sequential/encoder_conv3/Elu:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool3/MaxPool�
)sequential/encoder_dropout3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)sequential/encoder_dropout3/dropout/Const�
'sequential/encoder_dropout3/dropout/MulMul)sequential/encoder_pool3/MaxPool:output:02sequential/encoder_dropout3/dropout/Const:output:0*
T0*/
_output_shapes
:���������02)
'sequential/encoder_dropout3/dropout/Mul�
)sequential/encoder_dropout3/dropout/ShapeShape)sequential/encoder_pool3/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout3/dropout/Shape�
@sequential/encoder_dropout3/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout3/dropout/Shape:output:0*
T0*/
_output_shapes
:���������0*
dtype02B
@sequential/encoder_dropout3/dropout/random_uniform/RandomUniform�
2sequential/encoder_dropout3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2sequential/encoder_dropout3/dropout/GreaterEqual/y�
0sequential/encoder_dropout3/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout3/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������022
0sequential/encoder_dropout3/dropout/GreaterEqual�
(sequential/encoder_dropout3/dropout/CastCast4sequential/encoder_dropout3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������02*
(sequential/encoder_dropout3/dropout/Cast�
)sequential/encoder_dropout3/dropout/Mul_1Mul+sequential/encoder_dropout3/dropout/Mul:z:0,sequential/encoder_dropout3/dropout/Cast:y:0*
T0*/
_output_shapes
:���������02+
)sequential/encoder_dropout3/dropout/Mul_1�
.sequential/encoder_conv4/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype020
.sequential/encoder_conv4/Conv2D/ReadVariableOp�
sequential/encoder_conv4/Conv2DConv2D-sequential/encoder_dropout3/dropout/Mul_1:z:06sequential/encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2!
sequential/encoder_conv4/Conv2D�
/sequential/encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/encoder_conv4/BiasAdd/ReadVariableOp�
 sequential/encoder_conv4/BiasAddBiasAdd(sequential/encoder_conv4/Conv2D:output:07sequential/encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2"
 sequential/encoder_conv4/BiasAdd�
sequential/encoder_conv4/EluElu)sequential/encoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential/encoder_conv4/Elu�
 sequential/encoder_pool4/MaxPoolMaxPool*sequential/encoder_conv4/Elu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool4/MaxPool�
)sequential/encoder_dropout4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)sequential/encoder_dropout4/dropout/Const�
'sequential/encoder_dropout4/dropout/MulMul)sequential/encoder_pool4/MaxPool:output:02sequential/encoder_dropout4/dropout/Const:output:0*
T0*/
_output_shapes
:���������@2)
'sequential/encoder_dropout4/dropout/Mul�
)sequential/encoder_dropout4/dropout/ShapeShape)sequential/encoder_pool4/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout4/dropout/Shape�
@sequential/encoder_dropout4/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout4/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02B
@sequential/encoder_dropout4/dropout/random_uniform/RandomUniform�
2sequential/encoder_dropout4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2sequential/encoder_dropout4/dropout/GreaterEqual/y�
0sequential/encoder_dropout4/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout4/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@22
0sequential/encoder_dropout4/dropout/GreaterEqual�
(sequential/encoder_dropout4/dropout/CastCast4sequential/encoder_dropout4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2*
(sequential/encoder_dropout4/dropout/Cast�
)sequential/encoder_dropout4/dropout/Mul_1Mul+sequential/encoder_dropout4/dropout/Mul:z:0,sequential/encoder_dropout4/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2+
)sequential/encoder_dropout4/dropout/Mul_1�
.sequential/encoder_conv5/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype020
.sequential/encoder_conv5/Conv2D/ReadVariableOp�
sequential/encoder_conv5/Conv2DConv2D-sequential/encoder_dropout4/dropout/Mul_1:z:06sequential/encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2!
sequential/encoder_conv5/Conv2D�
/sequential/encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype021
/sequential/encoder_conv5/BiasAdd/ReadVariableOp�
 sequential/encoder_conv5/BiasAddBiasAdd(sequential/encoder_conv5/Conv2D:output:07sequential/encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2"
 sequential/encoder_conv5/BiasAdd�
sequential/encoder_conv5/EluElu)sequential/encoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:���������P2
sequential/encoder_conv5/Elu�
 sequential/encoder_pool5/MaxPoolMaxPool*sequential/encoder_conv5/Elu:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool5/MaxPool�
)sequential/encoder_dropout5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)sequential/encoder_dropout5/dropout/Const�
'sequential/encoder_dropout5/dropout/MulMul)sequential/encoder_pool5/MaxPool:output:02sequential/encoder_dropout5/dropout/Const:output:0*
T0*/
_output_shapes
:���������P2)
'sequential/encoder_dropout5/dropout/Mul�
)sequential/encoder_dropout5/dropout/ShapeShape)sequential/encoder_pool5/MaxPool:output:0*
T0*
_output_shapes
:2+
)sequential/encoder_dropout5/dropout/Shape�
@sequential/encoder_dropout5/dropout/random_uniform/RandomUniformRandomUniform2sequential/encoder_dropout5/dropout/Shape:output:0*
T0*/
_output_shapes
:���������P*
dtype02B
@sequential/encoder_dropout5/dropout/random_uniform/RandomUniform�
2sequential/encoder_dropout5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>24
2sequential/encoder_dropout5/dropout/GreaterEqual/y�
0sequential/encoder_dropout5/dropout/GreaterEqualGreaterEqualIsequential/encoder_dropout5/dropout/random_uniform/RandomUniform:output:0;sequential/encoder_dropout5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������P22
0sequential/encoder_dropout5/dropout/GreaterEqual�
(sequential/encoder_dropout5/dropout/CastCast4sequential/encoder_dropout5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������P2*
(sequential/encoder_dropout5/dropout/Cast�
)sequential/encoder_dropout5/dropout/Mul_1Mul+sequential/encoder_dropout5/dropout/Mul:z:0,sequential/encoder_dropout5/dropout/Cast:y:0*
T0*/
_output_shapes
:���������P2+
)sequential/encoder_dropout5/dropout/Mul_1�
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
sequential/flatten/Const�
sequential/flatten/ReshapeReshape-sequential/encoder_dropout5/dropout/Mul_1:z:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:����������
2
sequential/flatten/Reshape�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/BiasAdd�
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential/dense/Sigmoid�
Bsequential/random_flip/random_flip_left_right_1/control_dependencyIdentityinputs_1*
T0*
_class
loc:@inputs/1*1
_output_shapes
:�����������2D
Bsequential/random_flip/random_flip_left_right_1/control_dependency�
5sequential/random_flip/random_flip_left_right_1/ShapeShapeKsequential/random_flip/random_flip_left_right_1/control_dependency:output:0*
T0*
_output_shapes
:27
5sequential/random_flip/random_flip_left_right_1/Shape�
Csequential/random_flip/random_flip_left_right_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/random_flip/random_flip_left_right_1/strided_slice/stack�
Esequential/random_flip/random_flip_left_right_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential/random_flip/random_flip_left_right_1/strided_slice/stack_1�
Esequential/random_flip/random_flip_left_right_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential/random_flip/random_flip_left_right_1/strided_slice/stack_2�
=sequential/random_flip/random_flip_left_right_1/strided_sliceStridedSlice>sequential/random_flip/random_flip_left_right_1/Shape:output:0Lsequential/random_flip/random_flip_left_right_1/strided_slice/stack:output:0Nsequential/random_flip/random_flip_left_right_1/strided_slice/stack_1:output:0Nsequential/random_flip/random_flip_left_right_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=sequential/random_flip/random_flip_left_right_1/strided_slice�
Dsequential/random_flip/random_flip_left_right_1/random_uniform/shapePackFsequential/random_flip/random_flip_left_right_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2F
Dsequential/random_flip/random_flip_left_right_1/random_uniform/shape�
Bsequential/random_flip/random_flip_left_right_1/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2D
Bsequential/random_flip/random_flip_left_right_1/random_uniform/min�
Bsequential/random_flip/random_flip_left_right_1/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2D
Bsequential/random_flip/random_flip_left_right_1/random_uniform/max�
Lsequential/random_flip/random_flip_left_right_1/random_uniform/RandomUniformRandomUniformMsequential/random_flip/random_flip_left_right_1/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype02N
Lsequential/random_flip/random_flip_left_right_1/random_uniform/RandomUniform�
Bsequential/random_flip/random_flip_left_right_1/random_uniform/MulMulUsequential/random_flip/random_flip_left_right_1/random_uniform/RandomUniform:output:0Ksequential/random_flip/random_flip_left_right_1/random_uniform/max:output:0*
T0*#
_output_shapes
:���������2D
Bsequential/random_flip/random_flip_left_right_1/random_uniform/Mul�
?sequential/random_flip/random_flip_left_right_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?sequential/random_flip/random_flip_left_right_1/Reshape/shape/1�
?sequential/random_flip/random_flip_left_right_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?sequential/random_flip/random_flip_left_right_1/Reshape/shape/2�
?sequential/random_flip/random_flip_left_right_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?sequential/random_flip/random_flip_left_right_1/Reshape/shape/3�
=sequential/random_flip/random_flip_left_right_1/Reshape/shapePackFsequential/random_flip/random_flip_left_right_1/strided_slice:output:0Hsequential/random_flip/random_flip_left_right_1/Reshape/shape/1:output:0Hsequential/random_flip/random_flip_left_right_1/Reshape/shape/2:output:0Hsequential/random_flip/random_flip_left_right_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=sequential/random_flip/random_flip_left_right_1/Reshape/shape�
7sequential/random_flip/random_flip_left_right_1/ReshapeReshapeFsequential/random_flip/random_flip_left_right_1/random_uniform/Mul:z:0Fsequential/random_flip/random_flip_left_right_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������29
7sequential/random_flip/random_flip_left_right_1/Reshape�
5sequential/random_flip/random_flip_left_right_1/RoundRound@sequential/random_flip/random_flip_left_right_1/Reshape:output:0*
T0*/
_output_shapes
:���������27
5sequential/random_flip/random_flip_left_right_1/Round�
>sequential/random_flip/random_flip_left_right_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/random_flip/random_flip_left_right_1/ReverseV2/axis�
9sequential/random_flip/random_flip_left_right_1/ReverseV2	ReverseV2Ksequential/random_flip/random_flip_left_right_1/control_dependency:output:0Gsequential/random_flip/random_flip_left_right_1/ReverseV2/axis:output:0*
T0*1
_output_shapes
:�����������2;
9sequential/random_flip/random_flip_left_right_1/ReverseV2�
3sequential/random_flip/random_flip_left_right_1/mulMul9sequential/random_flip/random_flip_left_right_1/Round:y:0Bsequential/random_flip/random_flip_left_right_1/ReverseV2:output:0*
T0*1
_output_shapes
:�����������25
3sequential/random_flip/random_flip_left_right_1/mul�
5sequential/random_flip/random_flip_left_right_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?27
5sequential/random_flip/random_flip_left_right_1/sub/x�
3sequential/random_flip/random_flip_left_right_1/subSub>sequential/random_flip/random_flip_left_right_1/sub/x:output:09sequential/random_flip/random_flip_left_right_1/Round:y:0*
T0*/
_output_shapes
:���������25
3sequential/random_flip/random_flip_left_right_1/sub�
5sequential/random_flip/random_flip_left_right_1/mul_1Mul7sequential/random_flip/random_flip_left_right_1/sub:z:0Ksequential/random_flip/random_flip_left_right_1/control_dependency:output:0*
T0*1
_output_shapes
:�����������27
5sequential/random_flip/random_flip_left_right_1/mul_1�
3sequential/random_flip/random_flip_left_right_1/addAddV27sequential/random_flip/random_flip_left_right_1/mul:z:09sequential/random_flip/random_flip_left_right_1/mul_1:z:0*
T0*1
_output_shapes
:�����������25
3sequential/random_flip/random_flip_left_right_1/add�
sequential/random_zoom/Shape_1Shape7sequential/random_flip/random_flip_left_right_1/add:z:0*
T0*
_output_shapes
:2 
sequential/random_zoom/Shape_1�
,sequential/random_zoom/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential/random_zoom/strided_slice_3/stack�
.sequential/random_zoom/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_3/stack_1�
.sequential/random_zoom/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_3/stack_2�
&sequential/random_zoom/strided_slice_3StridedSlice'sequential/random_zoom/Shape_1:output:05sequential/random_zoom/strided_slice_3/stack:output:07sequential/random_zoom/strided_slice_3/stack_1:output:07sequential/random_zoom/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/random_zoom/strided_slice_3�
,sequential/random_zoom/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_zoom/strided_slice_4/stack�
.sequential/random_zoom/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_4/stack_1�
.sequential/random_zoom/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_4/stack_2�
&sequential/random_zoom/strided_slice_4StridedSlice'sequential/random_zoom/Shape_1:output:05sequential/random_zoom/strided_slice_4/stack:output:07sequential/random_zoom/strided_slice_4/stack_1:output:07sequential/random_zoom/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/random_zoom/strided_slice_4�
sequential/random_zoom/Cast_2Cast/sequential/random_zoom/strided_slice_4:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
sequential/random_zoom/Cast_2�
,sequential/random_zoom/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential/random_zoom/strided_slice_5/stack�
.sequential/random_zoom/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_5/stack_1�
.sequential/random_zoom/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/random_zoom/strided_slice_5/stack_2�
&sequential/random_zoom/strided_slice_5StridedSlice'sequential/random_zoom/Shape_1:output:05sequential/random_zoom/strided_slice_5/stack:output:07sequential/random_zoom/strided_slice_5/stack_1:output:07sequential/random_zoom/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential/random_zoom/strided_slice_5�
sequential/random_zoom/Cast_3Cast/sequential/random_zoom/strided_slice_5:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
sequential/random_zoom/Cast_3�
1sequential/random_zoom/stateful_uniform_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential/random_zoom/stateful_uniform_2/shape/1�
/sequential/random_zoom/stateful_uniform_2/shapePack/sequential/random_zoom/strided_slice_3:output:0:sequential/random_zoom/stateful_uniform_2/shape/1:output:0*
N*
T0*
_output_shapes
:21
/sequential/random_zoom/stateful_uniform_2/shape�
-sequential/random_zoom/stateful_uniform_2/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2/
-sequential/random_zoom/stateful_uniform_2/min�
-sequential/random_zoom/stateful_uniform_2/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2/
-sequential/random_zoom/stateful_uniform_2/max�
Csequential/random_zoom/stateful_uniform_2/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
Csequential/random_zoom/stateful_uniform_2/StatefulUniform/algorithm�
9sequential/random_zoom/stateful_uniform_2/StatefulUniformStatefulUniform@sequential_random_zoom_stateful_uniform_statefuluniform_resourceLsequential/random_zoom/stateful_uniform_2/StatefulUniform/algorithm:output:08sequential/random_zoom/stateful_uniform_2/shape:output:0:^sequential/random_zoom/stateful_uniform_1/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02;
9sequential/random_zoom/stateful_uniform_2/StatefulUniform�
-sequential/random_zoom/stateful_uniform_2/subSub6sequential/random_zoom/stateful_uniform_2/max:output:06sequential/random_zoom/stateful_uniform_2/min:output:0*
T0*
_output_shapes
: 2/
-sequential/random_zoom/stateful_uniform_2/sub�
-sequential/random_zoom/stateful_uniform_2/mulMulBsequential/random_zoom/stateful_uniform_2/StatefulUniform:output:01sequential/random_zoom/stateful_uniform_2/sub:z:0*
T0*'
_output_shapes
:���������2/
-sequential/random_zoom/stateful_uniform_2/mul�
)sequential/random_zoom/stateful_uniform_2Add1sequential/random_zoom/stateful_uniform_2/mul:z:06sequential/random_zoom/stateful_uniform_2/min:output:0*
T0*'
_output_shapes
:���������2+
)sequential/random_zoom/stateful_uniform_2�
1sequential/random_zoom/stateful_uniform_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential/random_zoom/stateful_uniform_3/shape/1�
/sequential/random_zoom/stateful_uniform_3/shapePack/sequential/random_zoom/strided_slice_3:output:0:sequential/random_zoom/stateful_uniform_3/shape/1:output:0*
N*
T0*
_output_shapes
:21
/sequential/random_zoom/stateful_uniform_3/shape�
-sequential/random_zoom/stateful_uniform_3/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2/
-sequential/random_zoom/stateful_uniform_3/min�
-sequential/random_zoom/stateful_uniform_3/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2/
-sequential/random_zoom/stateful_uniform_3/max�
Csequential/random_zoom/stateful_uniform_3/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
Csequential/random_zoom/stateful_uniform_3/StatefulUniform/algorithm�
9sequential/random_zoom/stateful_uniform_3/StatefulUniformStatefulUniform@sequential_random_zoom_stateful_uniform_statefuluniform_resourceLsequential/random_zoom/stateful_uniform_3/StatefulUniform/algorithm:output:08sequential/random_zoom/stateful_uniform_3/shape:output:0:^sequential/random_zoom/stateful_uniform_2/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02;
9sequential/random_zoom/stateful_uniform_3/StatefulUniform�
-sequential/random_zoom/stateful_uniform_3/subSub6sequential/random_zoom/stateful_uniform_3/max:output:06sequential/random_zoom/stateful_uniform_3/min:output:0*
T0*
_output_shapes
: 2/
-sequential/random_zoom/stateful_uniform_3/sub�
-sequential/random_zoom/stateful_uniform_3/mulMulBsequential/random_zoom/stateful_uniform_3/StatefulUniform:output:01sequential/random_zoom/stateful_uniform_3/sub:z:0*
T0*'
_output_shapes
:���������2/
-sequential/random_zoom/stateful_uniform_3/mul�
)sequential/random_zoom/stateful_uniform_3Add1sequential/random_zoom/stateful_uniform_3/mul:z:06sequential/random_zoom/stateful_uniform_3/min:output:0*
T0*'
_output_shapes
:���������2+
)sequential/random_zoom/stateful_uniform_3�
$sequential/random_zoom/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$sequential/random_zoom/concat_1/axis�
sequential/random_zoom/concat_1ConcatV2-sequential/random_zoom/stateful_uniform_3:z:0-sequential/random_zoom/stateful_uniform_2:z:0-sequential/random_zoom/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������2!
sequential/random_zoom/concat_1�
*sequential/random_zoom/zoom_matrix_1/ShapeShape(sequential/random_zoom/concat_1:output:0*
T0*
_output_shapes
:2,
*sequential/random_zoom/zoom_matrix_1/Shape�
8sequential/random_zoom/zoom_matrix_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential/random_zoom/zoom_matrix_1/strided_slice/stack�
:sequential/random_zoom/zoom_matrix_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/random_zoom/zoom_matrix_1/strided_slice/stack_1�
:sequential/random_zoom/zoom_matrix_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/random_zoom/zoom_matrix_1/strided_slice/stack_2�
2sequential/random_zoom/zoom_matrix_1/strided_sliceStridedSlice3sequential/random_zoom/zoom_matrix_1/Shape:output:0Asequential/random_zoom/zoom_matrix_1/strided_slice/stack:output:0Csequential/random_zoom/zoom_matrix_1/strided_slice/stack_1:output:0Csequential/random_zoom/zoom_matrix_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential/random_zoom/zoom_matrix_1/strided_slice�
*sequential/random_zoom/zoom_matrix_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2,
*sequential/random_zoom/zoom_matrix_1/sub/y�
(sequential/random_zoom/zoom_matrix_1/subSub!sequential/random_zoom/Cast_3:y:03sequential/random_zoom/zoom_matrix_1/sub/y:output:0*
T0*
_output_shapes
: 2*
(sequential/random_zoom/zoom_matrix_1/sub�
.sequential/random_zoom/zoom_matrix_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.sequential/random_zoom/zoom_matrix_1/truediv/y�
,sequential/random_zoom/zoom_matrix_1/truedivRealDiv,sequential/random_zoom/zoom_matrix_1/sub:z:07sequential/random_zoom/zoom_matrix_1/truediv/y:output:0*
T0*
_output_shapes
: 2.
,sequential/random_zoom/zoom_matrix_1/truediv�
:sequential/random_zoom/zoom_matrix_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2<
:sequential/random_zoom/zoom_matrix_1/strided_slice_1/stack�
<sequential/random_zoom/zoom_matrix_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_1/stack_1�
<sequential/random_zoom/zoom_matrix_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_1/stack_2�
4sequential/random_zoom/zoom_matrix_1/strided_slice_1StridedSlice(sequential/random_zoom/concat_1:output:0Csequential/random_zoom/zoom_matrix_1/strided_slice_1/stack:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_1/stack_1:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask26
4sequential/random_zoom/zoom_matrix_1/strided_slice_1�
,sequential/random_zoom/zoom_matrix_1/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,sequential/random_zoom/zoom_matrix_1/sub_1/x�
*sequential/random_zoom/zoom_matrix_1/sub_1Sub5sequential/random_zoom/zoom_matrix_1/sub_1/x:output:0=sequential/random_zoom/zoom_matrix_1/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2,
*sequential/random_zoom/zoom_matrix_1/sub_1�
(sequential/random_zoom/zoom_matrix_1/mulMul0sequential/random_zoom/zoom_matrix_1/truediv:z:0.sequential/random_zoom/zoom_matrix_1/sub_1:z:0*
T0*'
_output_shapes
:���������2*
(sequential/random_zoom/zoom_matrix_1/mul�
,sequential/random_zoom/zoom_matrix_1/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,sequential/random_zoom/zoom_matrix_1/sub_2/y�
*sequential/random_zoom/zoom_matrix_1/sub_2Sub!sequential/random_zoom/Cast_2:y:05sequential/random_zoom/zoom_matrix_1/sub_2/y:output:0*
T0*
_output_shapes
: 2,
*sequential/random_zoom/zoom_matrix_1/sub_2�
0sequential/random_zoom/zoom_matrix_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @22
0sequential/random_zoom/zoom_matrix_1/truediv_1/y�
.sequential/random_zoom/zoom_matrix_1/truediv_1RealDiv.sequential/random_zoom/zoom_matrix_1/sub_2:z:09sequential/random_zoom/zoom_matrix_1/truediv_1/y:output:0*
T0*
_output_shapes
: 20
.sequential/random_zoom/zoom_matrix_1/truediv_1�
:sequential/random_zoom/zoom_matrix_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2<
:sequential/random_zoom/zoom_matrix_1/strided_slice_2/stack�
<sequential/random_zoom/zoom_matrix_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_2/stack_1�
<sequential/random_zoom/zoom_matrix_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_2/stack_2�
4sequential/random_zoom/zoom_matrix_1/strided_slice_2StridedSlice(sequential/random_zoom/concat_1:output:0Csequential/random_zoom/zoom_matrix_1/strided_slice_2/stack:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_2/stack_1:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask26
4sequential/random_zoom/zoom_matrix_1/strided_slice_2�
,sequential/random_zoom/zoom_matrix_1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,sequential/random_zoom/zoom_matrix_1/sub_3/x�
*sequential/random_zoom/zoom_matrix_1/sub_3Sub5sequential/random_zoom/zoom_matrix_1/sub_3/x:output:0=sequential/random_zoom/zoom_matrix_1/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2,
*sequential/random_zoom/zoom_matrix_1/sub_3�
*sequential/random_zoom/zoom_matrix_1/mul_1Mul2sequential/random_zoom/zoom_matrix_1/truediv_1:z:0.sequential/random_zoom/zoom_matrix_1/sub_3:z:0*
T0*'
_output_shapes
:���������2,
*sequential/random_zoom/zoom_matrix_1/mul_1�
:sequential/random_zoom/zoom_matrix_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2<
:sequential/random_zoom/zoom_matrix_1/strided_slice_3/stack�
<sequential/random_zoom/zoom_matrix_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_3/stack_1�
<sequential/random_zoom/zoom_matrix_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_3/stack_2�
4sequential/random_zoom/zoom_matrix_1/strided_slice_3StridedSlice(sequential/random_zoom/concat_1:output:0Csequential/random_zoom/zoom_matrix_1/strided_slice_3/stack:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_3/stack_1:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask26
4sequential/random_zoom/zoom_matrix_1/strided_slice_3�
0sequential/random_zoom/zoom_matrix_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/random_zoom/zoom_matrix_1/zeros/mul/y�
.sequential/random_zoom/zoom_matrix_1/zeros/mulMul;sequential/random_zoom/zoom_matrix_1/strided_slice:output:09sequential/random_zoom/zoom_matrix_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 20
.sequential/random_zoom/zoom_matrix_1/zeros/mul�
1sequential/random_zoom/zoom_matrix_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�23
1sequential/random_zoom/zoom_matrix_1/zeros/Less/y�
/sequential/random_zoom/zoom_matrix_1/zeros/LessLess2sequential/random_zoom/zoom_matrix_1/zeros/mul:z:0:sequential/random_zoom/zoom_matrix_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 21
/sequential/random_zoom/zoom_matrix_1/zeros/Less�
3sequential/random_zoom/zoom_matrix_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential/random_zoom/zoom_matrix_1/zeros/packed/1�
1sequential/random_zoom/zoom_matrix_1/zeros/packedPack;sequential/random_zoom/zoom_matrix_1/strided_slice:output:0<sequential/random_zoom/zoom_matrix_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:23
1sequential/random_zoom/zoom_matrix_1/zeros/packed�
0sequential/random_zoom/zoom_matrix_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0sequential/random_zoom/zoom_matrix_1/zeros/Const�
*sequential/random_zoom/zoom_matrix_1/zerosFill:sequential/random_zoom/zoom_matrix_1/zeros/packed:output:09sequential/random_zoom/zoom_matrix_1/zeros/Const:output:0*
T0*'
_output_shapes
:���������2,
*sequential/random_zoom/zoom_matrix_1/zeros�
2sequential/random_zoom/zoom_matrix_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential/random_zoom/zoom_matrix_1/zeros_1/mul/y�
0sequential/random_zoom/zoom_matrix_1/zeros_1/mulMul;sequential/random_zoom/zoom_matrix_1/strided_slice:output:0;sequential/random_zoom/zoom_matrix_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential/random_zoom/zoom_matrix_1/zeros_1/mul�
3sequential/random_zoom/zoom_matrix_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�25
3sequential/random_zoom/zoom_matrix_1/zeros_1/Less/y�
1sequential/random_zoom/zoom_matrix_1/zeros_1/LessLess4sequential/random_zoom/zoom_matrix_1/zeros_1/mul:z:0<sequential/random_zoom/zoom_matrix_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 23
1sequential/random_zoom/zoom_matrix_1/zeros_1/Less�
5sequential/random_zoom/zoom_matrix_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential/random_zoom/zoom_matrix_1/zeros_1/packed/1�
3sequential/random_zoom/zoom_matrix_1/zeros_1/packedPack;sequential/random_zoom/zoom_matrix_1/strided_slice:output:0>sequential/random_zoom/zoom_matrix_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:25
3sequential/random_zoom/zoom_matrix_1/zeros_1/packed�
2sequential/random_zoom/zoom_matrix_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2sequential/random_zoom/zoom_matrix_1/zeros_1/Const�
,sequential/random_zoom/zoom_matrix_1/zeros_1Fill<sequential/random_zoom/zoom_matrix_1/zeros_1/packed:output:0;sequential/random_zoom/zoom_matrix_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2.
,sequential/random_zoom/zoom_matrix_1/zeros_1�
:sequential/random_zoom/zoom_matrix_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2<
:sequential/random_zoom/zoom_matrix_1/strided_slice_4/stack�
<sequential/random_zoom/zoom_matrix_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_4/stack_1�
<sequential/random_zoom/zoom_matrix_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2>
<sequential/random_zoom/zoom_matrix_1/strided_slice_4/stack_2�
4sequential/random_zoom/zoom_matrix_1/strided_slice_4StridedSlice(sequential/random_zoom/concat_1:output:0Csequential/random_zoom/zoom_matrix_1/strided_slice_4/stack:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_4/stack_1:output:0Esequential/random_zoom/zoom_matrix_1/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask26
4sequential/random_zoom/zoom_matrix_1/strided_slice_4�
2sequential/random_zoom/zoom_matrix_1/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential/random_zoom/zoom_matrix_1/zeros_2/mul/y�
0sequential/random_zoom/zoom_matrix_1/zeros_2/mulMul;sequential/random_zoom/zoom_matrix_1/strided_slice:output:0;sequential/random_zoom/zoom_matrix_1/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential/random_zoom/zoom_matrix_1/zeros_2/mul�
3sequential/random_zoom/zoom_matrix_1/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�25
3sequential/random_zoom/zoom_matrix_1/zeros_2/Less/y�
1sequential/random_zoom/zoom_matrix_1/zeros_2/LessLess4sequential/random_zoom/zoom_matrix_1/zeros_2/mul:z:0<sequential/random_zoom/zoom_matrix_1/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 23
1sequential/random_zoom/zoom_matrix_1/zeros_2/Less�
5sequential/random_zoom/zoom_matrix_1/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential/random_zoom/zoom_matrix_1/zeros_2/packed/1�
3sequential/random_zoom/zoom_matrix_1/zeros_2/packedPack;sequential/random_zoom/zoom_matrix_1/strided_slice:output:0>sequential/random_zoom/zoom_matrix_1/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:25
3sequential/random_zoom/zoom_matrix_1/zeros_2/packed�
2sequential/random_zoom/zoom_matrix_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2sequential/random_zoom/zoom_matrix_1/zeros_2/Const�
,sequential/random_zoom/zoom_matrix_1/zeros_2Fill<sequential/random_zoom/zoom_matrix_1/zeros_2/packed:output:0;sequential/random_zoom/zoom_matrix_1/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2.
,sequential/random_zoom/zoom_matrix_1/zeros_2�
0sequential/random_zoom/zoom_matrix_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/random_zoom/zoom_matrix_1/concat/axis�
+sequential/random_zoom/zoom_matrix_1/concatConcatV2=sequential/random_zoom/zoom_matrix_1/strided_slice_3:output:03sequential/random_zoom/zoom_matrix_1/zeros:output:0,sequential/random_zoom/zoom_matrix_1/mul:z:05sequential/random_zoom/zoom_matrix_1/zeros_1:output:0=sequential/random_zoom/zoom_matrix_1/strided_slice_4:output:0.sequential/random_zoom/zoom_matrix_1/mul_1:z:05sequential/random_zoom/zoom_matrix_1/zeros_2:output:09sequential/random_zoom/zoom_matrix_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2-
+sequential/random_zoom/zoom_matrix_1/concat�
(sequential/random_zoom/transform_1/ShapeShape7sequential/random_flip/random_flip_left_right_1/add:z:0*
T0*
_output_shapes
:2*
(sequential/random_zoom/transform_1/Shape�
6sequential/random_zoom/transform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6sequential/random_zoom/transform_1/strided_slice/stack�
8sequential/random_zoom/transform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential/random_zoom/transform_1/strided_slice/stack_1�
8sequential/random_zoom/transform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential/random_zoom/transform_1/strided_slice/stack_2�
0sequential/random_zoom/transform_1/strided_sliceStridedSlice1sequential/random_zoom/transform_1/Shape:output:0?sequential/random_zoom/transform_1/strided_slice/stack:output:0Asequential/random_zoom/transform_1/strided_slice/stack_1:output:0Asequential/random_zoom/transform_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0sequential/random_zoom/transform_1/strided_slice�
=sequential/random_zoom/transform_1/ImageProjectiveTransformV2ImageProjectiveTransformV27sequential/random_flip/random_flip_left_right_1/add:z:04sequential/random_zoom/zoom_matrix_1/concat:output:09sequential/random_zoom/transform_1/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2?
=sequential/random_zoom/transform_1/ImageProjectiveTransformV2�
%sequential/random_translation/Shape_1ShapeRsequential/random_zoom/transform_1/ImageProjectiveTransformV2:transformed_images:0*
T0*
_output_shapes
:2'
%sequential/random_translation/Shape_1�
3sequential/random_translation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/random_translation/strided_slice_3/stack�
5sequential/random_translation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_3/stack_1�
5sequential/random_translation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_3/stack_2�
-sequential/random_translation/strided_slice_3StridedSlice.sequential/random_translation/Shape_1:output:0<sequential/random_translation/strided_slice_3/stack:output:0>sequential/random_translation/strided_slice_3/stack_1:output:0>sequential/random_translation/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/random_translation/strided_slice_3�
3sequential/random_translation/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/random_translation/strided_slice_4/stack�
5sequential/random_translation/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_4/stack_1�
5sequential/random_translation/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_4/stack_2�
-sequential/random_translation/strided_slice_4StridedSlice.sequential/random_translation/Shape_1:output:0<sequential/random_translation/strided_slice_4/stack:output:0>sequential/random_translation/strided_slice_4/stack_1:output:0>sequential/random_translation/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/random_translation/strided_slice_4�
$sequential/random_translation/Cast_2Cast6sequential/random_translation/strided_slice_4:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$sequential/random_translation/Cast_2�
3sequential/random_translation/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential/random_translation/strided_slice_5/stack�
5sequential/random_translation/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_5/stack_1�
5sequential/random_translation/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/random_translation/strided_slice_5/stack_2�
-sequential/random_translation/strided_slice_5StridedSlice.sequential/random_translation/Shape_1:output:0<sequential/random_translation/strided_slice_5/stack:output:0>sequential/random_translation/strided_slice_5/stack_1:output:0>sequential/random_translation/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/random_translation/strided_slice_5�
$sequential/random_translation/Cast_3Cast6sequential/random_translation/strided_slice_5:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$sequential/random_translation/Cast_3�
8sequential/random_translation/stateful_uniform_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8sequential/random_translation/stateful_uniform_2/shape/1�
6sequential/random_translation/stateful_uniform_2/shapePack6sequential/random_translation/strided_slice_3:output:0Asequential/random_translation/stateful_uniform_2/shape/1:output:0*
N*
T0*
_output_shapes
:28
6sequential/random_translation/stateful_uniform_2/shape�
4sequential/random_translation/stateful_uniform_2/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�26
4sequential/random_translation/stateful_uniform_2/min�
4sequential/random_translation/stateful_uniform_2/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/random_translation/stateful_uniform_2/max�
Jsequential/random_translation/stateful_uniform_2/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2L
Jsequential/random_translation/stateful_uniform_2/StatefulUniform/algorithm�
@sequential/random_translation/stateful_uniform_2/StatefulUniformStatefulUniformGsequential_random_translation_stateful_uniform_statefuluniform_resourceSsequential/random_translation/stateful_uniform_2/StatefulUniform/algorithm:output:0?sequential/random_translation/stateful_uniform_2/shape:output:0A^sequential/random_translation/stateful_uniform_1/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02B
@sequential/random_translation/stateful_uniform_2/StatefulUniform�
4sequential/random_translation/stateful_uniform_2/subSub=sequential/random_translation/stateful_uniform_2/max:output:0=sequential/random_translation/stateful_uniform_2/min:output:0*
T0*
_output_shapes
: 26
4sequential/random_translation/stateful_uniform_2/sub�
4sequential/random_translation/stateful_uniform_2/mulMulIsequential/random_translation/stateful_uniform_2/StatefulUniform:output:08sequential/random_translation/stateful_uniform_2/sub:z:0*
T0*'
_output_shapes
:���������26
4sequential/random_translation/stateful_uniform_2/mul�
0sequential/random_translation/stateful_uniform_2Add8sequential/random_translation/stateful_uniform_2/mul:z:0=sequential/random_translation/stateful_uniform_2/min:output:0*
T0*'
_output_shapes
:���������22
0sequential/random_translation/stateful_uniform_2�
#sequential/random_translation/mul_2Mul4sequential/random_translation/stateful_uniform_2:z:0(sequential/random_translation/Cast_2:y:0*
T0*'
_output_shapes
:���������2%
#sequential/random_translation/mul_2�
8sequential/random_translation/stateful_uniform_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8sequential/random_translation/stateful_uniform_3/shape/1�
6sequential/random_translation/stateful_uniform_3/shapePack6sequential/random_translation/strided_slice_3:output:0Asequential/random_translation/stateful_uniform_3/shape/1:output:0*
N*
T0*
_output_shapes
:28
6sequential/random_translation/stateful_uniform_3/shape�
4sequential/random_translation/stateful_uniform_3/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�26
4sequential/random_translation/stateful_uniform_3/min�
4sequential/random_translation/stateful_uniform_3/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/random_translation/stateful_uniform_3/max�
Jsequential/random_translation/stateful_uniform_3/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2L
Jsequential/random_translation/stateful_uniform_3/StatefulUniform/algorithm�
@sequential/random_translation/stateful_uniform_3/StatefulUniformStatefulUniformGsequential_random_translation_stateful_uniform_statefuluniform_resourceSsequential/random_translation/stateful_uniform_3/StatefulUniform/algorithm:output:0?sequential/random_translation/stateful_uniform_3/shape:output:0A^sequential/random_translation/stateful_uniform_2/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02B
@sequential/random_translation/stateful_uniform_3/StatefulUniform�
4sequential/random_translation/stateful_uniform_3/subSub=sequential/random_translation/stateful_uniform_3/max:output:0=sequential/random_translation/stateful_uniform_3/min:output:0*
T0*
_output_shapes
: 26
4sequential/random_translation/stateful_uniform_3/sub�
4sequential/random_translation/stateful_uniform_3/mulMulIsequential/random_translation/stateful_uniform_3/StatefulUniform:output:08sequential/random_translation/stateful_uniform_3/sub:z:0*
T0*'
_output_shapes
:���������26
4sequential/random_translation/stateful_uniform_3/mul�
0sequential/random_translation/stateful_uniform_3Add8sequential/random_translation/stateful_uniform_3/mul:z:0=sequential/random_translation/stateful_uniform_3/min:output:0*
T0*'
_output_shapes
:���������22
0sequential/random_translation/stateful_uniform_3�
#sequential/random_translation/mul_3Mul4sequential/random_translation/stateful_uniform_3:z:0(sequential/random_translation/Cast_3:y:0*
T0*'
_output_shapes
:���������2%
#sequential/random_translation/mul_3�
+sequential/random_translation/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential/random_translation/concat_1/axis�
&sequential/random_translation/concat_1ConcatV2'sequential/random_translation/mul_3:z:0'sequential/random_translation/mul_2:z:04sequential/random_translation/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������2(
&sequential/random_translation/concat_1�
8sequential/random_translation/translation_matrix_1/ShapeShape/sequential/random_translation/concat_1:output:0*
T0*
_output_shapes
:2:
8sequential/random_translation/translation_matrix_1/Shape�
Fsequential/random_translation/translation_matrix_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential/random_translation/translation_matrix_1/strided_slice/stack�
Hsequential/random_translation/translation_matrix_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential/random_translation/translation_matrix_1/strided_slice/stack_1�
Hsequential/random_translation/translation_matrix_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential/random_translation/translation_matrix_1/strided_slice/stack_2�
@sequential/random_translation/translation_matrix_1/strided_sliceStridedSliceAsequential/random_translation/translation_matrix_1/Shape:output:0Osequential/random_translation/translation_matrix_1/strided_slice/stack:output:0Qsequential/random_translation/translation_matrix_1/strided_slice/stack_1:output:0Qsequential/random_translation/translation_matrix_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@sequential/random_translation/translation_matrix_1/strided_slice�
=sequential/random_translation/translation_matrix_1/ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/random_translation/translation_matrix_1/ones/mul/y�
;sequential/random_translation/translation_matrix_1/ones/mulMulIsequential/random_translation/translation_matrix_1/strided_slice:output:0Fsequential/random_translation/translation_matrix_1/ones/mul/y:output:0*
T0*
_output_shapes
: 2=
;sequential/random_translation/translation_matrix_1/ones/mul�
>sequential/random_translation/translation_matrix_1/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2@
>sequential/random_translation/translation_matrix_1/ones/Less/y�
<sequential/random_translation/translation_matrix_1/ones/LessLess?sequential/random_translation/translation_matrix_1/ones/mul:z:0Gsequential/random_translation/translation_matrix_1/ones/Less/y:output:0*
T0*
_output_shapes
: 2>
<sequential/random_translation/translation_matrix_1/ones/Less�
@sequential/random_translation/translation_matrix_1/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2B
@sequential/random_translation/translation_matrix_1/ones/packed/1�
>sequential/random_translation/translation_matrix_1/ones/packedPackIsequential/random_translation/translation_matrix_1/strided_slice:output:0Isequential/random_translation/translation_matrix_1/ones/packed/1:output:0*
N*
T0*
_output_shapes
:2@
>sequential/random_translation/translation_matrix_1/ones/packed�
=sequential/random_translation/translation_matrix_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2?
=sequential/random_translation/translation_matrix_1/ones/Const�
7sequential/random_translation/translation_matrix_1/onesFillGsequential/random_translation/translation_matrix_1/ones/packed:output:0Fsequential/random_translation/translation_matrix_1/ones/Const:output:0*
T0*'
_output_shapes
:���������29
7sequential/random_translation/translation_matrix_1/ones�
>sequential/random_translation/translation_matrix_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/random_translation/translation_matrix_1/zeros/mul/y�
<sequential/random_translation/translation_matrix_1/zeros/mulMulIsequential/random_translation/translation_matrix_1/strided_slice:output:0Gsequential/random_translation/translation_matrix_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential/random_translation/translation_matrix_1/zeros/mul�
?sequential/random_translation/translation_matrix_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2A
?sequential/random_translation/translation_matrix_1/zeros/Less/y�
=sequential/random_translation/translation_matrix_1/zeros/LessLess@sequential/random_translation/translation_matrix_1/zeros/mul:z:0Hsequential/random_translation/translation_matrix_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2?
=sequential/random_translation/translation_matrix_1/zeros/Less�
Asequential/random_translation/translation_matrix_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential/random_translation/translation_matrix_1/zeros/packed/1�
?sequential/random_translation/translation_matrix_1/zeros/packedPackIsequential/random_translation/translation_matrix_1/strided_slice:output:0Jsequential/random_translation/translation_matrix_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2A
?sequential/random_translation/translation_matrix_1/zeros/packed�
>sequential/random_translation/translation_matrix_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>sequential/random_translation/translation_matrix_1/zeros/Const�
8sequential/random_translation/translation_matrix_1/zerosFillHsequential/random_translation/translation_matrix_1/zeros/packed:output:0Gsequential/random_translation/translation_matrix_1/zeros/Const:output:0*
T0*'
_output_shapes
:���������2:
8sequential/random_translation/translation_matrix_1/zeros�
Hsequential/random_translation/translation_matrix_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2J
Hsequential/random_translation/translation_matrix_1/strided_slice_1/stack�
Jsequential/random_translation/translation_matrix_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2L
Jsequential/random_translation/translation_matrix_1/strided_slice_1/stack_1�
Jsequential/random_translation/translation_matrix_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2L
Jsequential/random_translation/translation_matrix_1/strided_slice_1/stack_2�
Bsequential/random_translation/translation_matrix_1/strided_slice_1StridedSlice/sequential/random_translation/concat_1:output:0Qsequential/random_translation/translation_matrix_1/strided_slice_1/stack:output:0Ssequential/random_translation/translation_matrix_1/strided_slice_1/stack_1:output:0Ssequential/random_translation/translation_matrix_1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2D
Bsequential/random_translation/translation_matrix_1/strided_slice_1�
6sequential/random_translation/translation_matrix_1/NegNegKsequential/random_translation/translation_matrix_1/strided_slice_1:output:0*
T0*'
_output_shapes
:���������28
6sequential/random_translation/translation_matrix_1/Neg�
@sequential/random_translation/translation_matrix_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential/random_translation/translation_matrix_1/zeros_1/mul/y�
>sequential/random_translation/translation_matrix_1/zeros_1/mulMulIsequential/random_translation/translation_matrix_1/strided_slice:output:0Isequential/random_translation/translation_matrix_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential/random_translation/translation_matrix_1/zeros_1/mul�
Asequential/random_translation/translation_matrix_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2C
Asequential/random_translation/translation_matrix_1/zeros_1/Less/y�
?sequential/random_translation/translation_matrix_1/zeros_1/LessLessBsequential/random_translation/translation_matrix_1/zeros_1/mul:z:0Jsequential/random_translation/translation_matrix_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential/random_translation/translation_matrix_1/zeros_1/Less�
Csequential/random_translation/translation_matrix_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2E
Csequential/random_translation/translation_matrix_1/zeros_1/packed/1�
Asequential/random_translation/translation_matrix_1/zeros_1/packedPackIsequential/random_translation/translation_matrix_1/strided_slice:output:0Lsequential/random_translation/translation_matrix_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential/random_translation/translation_matrix_1/zeros_1/packed�
@sequential/random_translation/translation_matrix_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@sequential/random_translation/translation_matrix_1/zeros_1/Const�
:sequential/random_translation/translation_matrix_1/zeros_1FillJsequential/random_translation/translation_matrix_1/zeros_1/packed:output:0Isequential/random_translation/translation_matrix_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2<
:sequential/random_translation/translation_matrix_1/zeros_1�
?sequential/random_translation/translation_matrix_1/ones_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?sequential/random_translation/translation_matrix_1/ones_1/mul/y�
=sequential/random_translation/translation_matrix_1/ones_1/mulMulIsequential/random_translation/translation_matrix_1/strided_slice:output:0Hsequential/random_translation/translation_matrix_1/ones_1/mul/y:output:0*
T0*
_output_shapes
: 2?
=sequential/random_translation/translation_matrix_1/ones_1/mul�
@sequential/random_translation/translation_matrix_1/ones_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2B
@sequential/random_translation/translation_matrix_1/ones_1/Less/y�
>sequential/random_translation/translation_matrix_1/ones_1/LessLessAsequential/random_translation/translation_matrix_1/ones_1/mul:z:0Isequential/random_translation/translation_matrix_1/ones_1/Less/y:output:0*
T0*
_output_shapes
: 2@
>sequential/random_translation/translation_matrix_1/ones_1/Less�
Bsequential/random_translation/translation_matrix_1/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bsequential/random_translation/translation_matrix_1/ones_1/packed/1�
@sequential/random_translation/translation_matrix_1/ones_1/packedPackIsequential/random_translation/translation_matrix_1/strided_slice:output:0Ksequential/random_translation/translation_matrix_1/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@sequential/random_translation/translation_matrix_1/ones_1/packed�
?sequential/random_translation/translation_matrix_1/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2A
?sequential/random_translation/translation_matrix_1/ones_1/Const�
9sequential/random_translation/translation_matrix_1/ones_1FillIsequential/random_translation/translation_matrix_1/ones_1/packed:output:0Hsequential/random_translation/translation_matrix_1/ones_1/Const:output:0*
T0*'
_output_shapes
:���������2;
9sequential/random_translation/translation_matrix_1/ones_1�
Hsequential/random_translation/translation_matrix_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2J
Hsequential/random_translation/translation_matrix_1/strided_slice_2/stack�
Jsequential/random_translation/translation_matrix_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2L
Jsequential/random_translation/translation_matrix_1/strided_slice_2/stack_1�
Jsequential/random_translation/translation_matrix_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2L
Jsequential/random_translation/translation_matrix_1/strided_slice_2/stack_2�
Bsequential/random_translation/translation_matrix_1/strided_slice_2StridedSlice/sequential/random_translation/concat_1:output:0Qsequential/random_translation/translation_matrix_1/strided_slice_2/stack:output:0Ssequential/random_translation/translation_matrix_1/strided_slice_2/stack_1:output:0Ssequential/random_translation/translation_matrix_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2D
Bsequential/random_translation/translation_matrix_1/strided_slice_2�
8sequential/random_translation/translation_matrix_1/Neg_1NegKsequential/random_translation/translation_matrix_1/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2:
8sequential/random_translation/translation_matrix_1/Neg_1�
@sequential/random_translation/translation_matrix_1/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@sequential/random_translation/translation_matrix_1/zeros_2/mul/y�
>sequential/random_translation/translation_matrix_1/zeros_2/mulMulIsequential/random_translation/translation_matrix_1/strided_slice:output:0Isequential/random_translation/translation_matrix_1/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2@
>sequential/random_translation/translation_matrix_1/zeros_2/mul�
Asequential/random_translation/translation_matrix_1/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2C
Asequential/random_translation/translation_matrix_1/zeros_2/Less/y�
?sequential/random_translation/translation_matrix_1/zeros_2/LessLessBsequential/random_translation/translation_matrix_1/zeros_2/mul:z:0Jsequential/random_translation/translation_matrix_1/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2A
?sequential/random_translation/translation_matrix_1/zeros_2/Less�
Csequential/random_translation/translation_matrix_1/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2E
Csequential/random_translation/translation_matrix_1/zeros_2/packed/1�
Asequential/random_translation/translation_matrix_1/zeros_2/packedPackIsequential/random_translation/translation_matrix_1/strided_slice:output:0Lsequential/random_translation/translation_matrix_1/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2C
Asequential/random_translation/translation_matrix_1/zeros_2/packed�
@sequential/random_translation/translation_matrix_1/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@sequential/random_translation/translation_matrix_1/zeros_2/Const�
:sequential/random_translation/translation_matrix_1/zeros_2FillJsequential/random_translation/translation_matrix_1/zeros_2/packed:output:0Isequential/random_translation/translation_matrix_1/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2<
:sequential/random_translation/translation_matrix_1/zeros_2�
>sequential/random_translation/translation_matrix_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/random_translation/translation_matrix_1/concat/axis�
9sequential/random_translation/translation_matrix_1/concatConcatV2@sequential/random_translation/translation_matrix_1/ones:output:0Asequential/random_translation/translation_matrix_1/zeros:output:0:sequential/random_translation/translation_matrix_1/Neg:y:0Csequential/random_translation/translation_matrix_1/zeros_1:output:0Bsequential/random_translation/translation_matrix_1/ones_1:output:0<sequential/random_translation/translation_matrix_1/Neg_1:y:0Csequential/random_translation/translation_matrix_1/zeros_2:output:0Gsequential/random_translation/translation_matrix_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2;
9sequential/random_translation/translation_matrix_1/concat�
/sequential/random_translation/transform_1/ShapeShapeRsequential/random_zoom/transform_1/ImageProjectiveTransformV2:transformed_images:0*
T0*
_output_shapes
:21
/sequential/random_translation/transform_1/Shape�
=sequential/random_translation/transform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential/random_translation/transform_1/strided_slice/stack�
?sequential/random_translation/transform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?sequential/random_translation/transform_1/strided_slice/stack_1�
?sequential/random_translation/transform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?sequential/random_translation/transform_1/strided_slice/stack_2�
7sequential/random_translation/transform_1/strided_sliceStridedSlice8sequential/random_translation/transform_1/Shape:output:0Fsequential/random_translation/transform_1/strided_slice/stack:output:0Hsequential/random_translation/transform_1/strided_slice/stack_1:output:0Hsequential/random_translation/transform_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:29
7sequential/random_translation/transform_1/strided_slice�
Dsequential/random_translation/transform_1/ImageProjectiveTransformV2ImageProjectiveTransformV2Rsequential/random_zoom/transform_1/ImageProjectiveTransformV2:transformed_images:0Bsequential/random_translation/translation_matrix_1/concat:output:0@sequential/random_translation/transform_1/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2F
Dsequential/random_translation/transform_1/ImageProjectiveTransformV2�
0sequential/encoder_conv1/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0sequential/encoder_conv1/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv1/Conv2D_1Conv2DYsequential/random_translation/transform_1/ImageProjectiveTransformV2:transformed_images:08sequential/encoder_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2#
!sequential/encoder_conv1/Conv2D_1�
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv1/BiasAdd_1BiasAdd*sequential/encoder_conv1/Conv2D_1:output:09sequential/encoder_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2$
"sequential/encoder_conv1/BiasAdd_1�
sequential/encoder_conv1/Elu_1Elu+sequential/encoder_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:�����������2 
sequential/encoder_conv1/Elu_1�
"sequential/encoder_pool1/MaxPool_1MaxPool,sequential/encoder_conv1/Elu_1:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool1/MaxPool_1�
+sequential/encoder_dropout1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+sequential/encoder_dropout1/dropout_1/Const�
)sequential/encoder_dropout1/dropout_1/MulMul+sequential/encoder_pool1/MaxPool_1:output:04sequential/encoder_dropout1/dropout_1/Const:output:0*
T0*/
_output_shapes
:���������@@2+
)sequential/encoder_dropout1/dropout_1/Mul�
+sequential/encoder_dropout1/dropout_1/ShapeShape+sequential/encoder_pool1/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout1/dropout_1/Shape�
Bsequential/encoder_dropout1/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout1/dropout_1/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype02D
Bsequential/encoder_dropout1/dropout_1/random_uniform/RandomUniform�
4sequential/encoder_dropout1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/encoder_dropout1/dropout_1/GreaterEqual/y�
2sequential/encoder_dropout1/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout1/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout1/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@@24
2sequential/encoder_dropout1/dropout_1/GreaterEqual�
*sequential/encoder_dropout1/dropout_1/CastCast6sequential/encoder_dropout1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2,
*sequential/encoder_dropout1/dropout_1/Cast�
+sequential/encoder_dropout1/dropout_1/Mul_1Mul-sequential/encoder_dropout1/dropout_1/Mul:z:0.sequential/encoder_dropout1/dropout_1/Cast:y:0*
T0*/
_output_shapes
:���������@@2-
+sequential/encoder_dropout1/dropout_1/Mul_1�
0sequential/encoder_conv2/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0sequential/encoder_conv2/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv2/Conv2D_1Conv2D/sequential/encoder_dropout1/dropout_1/Mul_1:z:08sequential/encoder_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2#
!sequential/encoder_conv2/Conv2D_1�
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv2/BiasAdd_1BiasAdd*sequential/encoder_conv2/Conv2D_1:output:09sequential/encoder_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2$
"sequential/encoder_conv2/BiasAdd_1�
sequential/encoder_conv2/Elu_1Elu+sequential/encoder_conv2/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������@@ 2 
sequential/encoder_conv2/Elu_1�
"sequential/encoder_pool2/MaxPool_1MaxPool,sequential/encoder_conv2/Elu_1:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool2/MaxPool_1�
+sequential/encoder_dropout2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+sequential/encoder_dropout2/dropout_1/Const�
)sequential/encoder_dropout2/dropout_1/MulMul+sequential/encoder_pool2/MaxPool_1:output:04sequential/encoder_dropout2/dropout_1/Const:output:0*
T0*/
_output_shapes
:���������   2+
)sequential/encoder_dropout2/dropout_1/Mul�
+sequential/encoder_dropout2/dropout_1/ShapeShape+sequential/encoder_pool2/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout2/dropout_1/Shape�
Bsequential/encoder_dropout2/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout2/dropout_1/Shape:output:0*
T0*/
_output_shapes
:���������   *
dtype02D
Bsequential/encoder_dropout2/dropout_1/random_uniform/RandomUniform�
4sequential/encoder_dropout2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/encoder_dropout2/dropout_1/GreaterEqual/y�
2sequential/encoder_dropout2/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout2/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout2/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������   24
2sequential/encoder_dropout2/dropout_1/GreaterEqual�
*sequential/encoder_dropout2/dropout_1/CastCast6sequential/encoder_dropout2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������   2,
*sequential/encoder_dropout2/dropout_1/Cast�
+sequential/encoder_dropout2/dropout_1/Mul_1Mul-sequential/encoder_dropout2/dropout_1/Mul:z:0.sequential/encoder_dropout2/dropout_1/Cast:y:0*
T0*/
_output_shapes
:���������   2-
+sequential/encoder_dropout2/dropout_1/Mul_1�
0sequential/encoder_conv3/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype022
0sequential/encoder_conv3/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv3/Conv2D_1Conv2D/sequential/encoder_dropout2/dropout_1/Mul_1:z:08sequential/encoder_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2#
!sequential/encoder_conv3/Conv2D_1�
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv3/BiasAdd_1BiasAdd*sequential/encoder_conv3/Conv2D_1:output:09sequential/encoder_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02$
"sequential/encoder_conv3/BiasAdd_1�
sequential/encoder_conv3/Elu_1Elu+sequential/encoder_conv3/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������  02 
sequential/encoder_conv3/Elu_1�
"sequential/encoder_pool3/MaxPool_1MaxPool,sequential/encoder_conv3/Elu_1:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool3/MaxPool_1�
+sequential/encoder_dropout3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+sequential/encoder_dropout3/dropout_1/Const�
)sequential/encoder_dropout3/dropout_1/MulMul+sequential/encoder_pool3/MaxPool_1:output:04sequential/encoder_dropout3/dropout_1/Const:output:0*
T0*/
_output_shapes
:���������02+
)sequential/encoder_dropout3/dropout_1/Mul�
+sequential/encoder_dropout3/dropout_1/ShapeShape+sequential/encoder_pool3/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout3/dropout_1/Shape�
Bsequential/encoder_dropout3/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout3/dropout_1/Shape:output:0*
T0*/
_output_shapes
:���������0*
dtype02D
Bsequential/encoder_dropout3/dropout_1/random_uniform/RandomUniform�
4sequential/encoder_dropout3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/encoder_dropout3/dropout_1/GreaterEqual/y�
2sequential/encoder_dropout3/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout3/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout3/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������024
2sequential/encoder_dropout3/dropout_1/GreaterEqual�
*sequential/encoder_dropout3/dropout_1/CastCast6sequential/encoder_dropout3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������02,
*sequential/encoder_dropout3/dropout_1/Cast�
+sequential/encoder_dropout3/dropout_1/Mul_1Mul-sequential/encoder_dropout3/dropout_1/Mul:z:0.sequential/encoder_dropout3/dropout_1/Cast:y:0*
T0*/
_output_shapes
:���������02-
+sequential/encoder_dropout3/dropout_1/Mul_1�
0sequential/encoder_conv4/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype022
0sequential/encoder_conv4/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv4/Conv2D_1Conv2D/sequential/encoder_dropout3/dropout_1/Mul_1:z:08sequential/encoder_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!sequential/encoder_conv4/Conv2D_1�
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv4/BiasAdd_1BiasAdd*sequential/encoder_conv4/Conv2D_1:output:09sequential/encoder_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2$
"sequential/encoder_conv4/BiasAdd_1�
sequential/encoder_conv4/Elu_1Elu+sequential/encoder_conv4/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������@2 
sequential/encoder_conv4/Elu_1�
"sequential/encoder_pool4/MaxPool_1MaxPool,sequential/encoder_conv4/Elu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool4/MaxPool_1�
+sequential/encoder_dropout4/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+sequential/encoder_dropout4/dropout_1/Const�
)sequential/encoder_dropout4/dropout_1/MulMul+sequential/encoder_pool4/MaxPool_1:output:04sequential/encoder_dropout4/dropout_1/Const:output:0*
T0*/
_output_shapes
:���������@2+
)sequential/encoder_dropout4/dropout_1/Mul�
+sequential/encoder_dropout4/dropout_1/ShapeShape+sequential/encoder_pool4/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout4/dropout_1/Shape�
Bsequential/encoder_dropout4/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout4/dropout_1/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02D
Bsequential/encoder_dropout4/dropout_1/random_uniform/RandomUniform�
4sequential/encoder_dropout4/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/encoder_dropout4/dropout_1/GreaterEqual/y�
2sequential/encoder_dropout4/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout4/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout4/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@24
2sequential/encoder_dropout4/dropout_1/GreaterEqual�
*sequential/encoder_dropout4/dropout_1/CastCast6sequential/encoder_dropout4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2,
*sequential/encoder_dropout4/dropout_1/Cast�
+sequential/encoder_dropout4/dropout_1/Mul_1Mul-sequential/encoder_dropout4/dropout_1/Mul:z:0.sequential/encoder_dropout4/dropout_1/Cast:y:0*
T0*/
_output_shapes
:���������@2-
+sequential/encoder_dropout4/dropout_1/Mul_1�
0sequential/encoder_conv5/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype022
0sequential/encoder_conv5/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv5/Conv2D_1Conv2D/sequential/encoder_dropout4/dropout_1/Mul_1:z:08sequential/encoder_conv5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2#
!sequential/encoder_conv5/Conv2D_1�
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype023
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv5/BiasAdd_1BiasAdd*sequential/encoder_conv5/Conv2D_1:output:09sequential/encoder_conv5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2$
"sequential/encoder_conv5/BiasAdd_1�
sequential/encoder_conv5/Elu_1Elu+sequential/encoder_conv5/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������P2 
sequential/encoder_conv5/Elu_1�
"sequential/encoder_pool5/MaxPool_1MaxPool,sequential/encoder_conv5/Elu_1:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool5/MaxPool_1�
+sequential/encoder_dropout5/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+sequential/encoder_dropout5/dropout_1/Const�
)sequential/encoder_dropout5/dropout_1/MulMul+sequential/encoder_pool5/MaxPool_1:output:04sequential/encoder_dropout5/dropout_1/Const:output:0*
T0*/
_output_shapes
:���������P2+
)sequential/encoder_dropout5/dropout_1/Mul�
+sequential/encoder_dropout5/dropout_1/ShapeShape+sequential/encoder_pool5/MaxPool_1:output:0*
T0*
_output_shapes
:2-
+sequential/encoder_dropout5/dropout_1/Shape�
Bsequential/encoder_dropout5/dropout_1/random_uniform/RandomUniformRandomUniform4sequential/encoder_dropout5/dropout_1/Shape:output:0*
T0*/
_output_shapes
:���������P*
dtype02D
Bsequential/encoder_dropout5/dropout_1/random_uniform/RandomUniform�
4sequential/encoder_dropout5/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>26
4sequential/encoder_dropout5/dropout_1/GreaterEqual/y�
2sequential/encoder_dropout5/dropout_1/GreaterEqualGreaterEqualKsequential/encoder_dropout5/dropout_1/random_uniform/RandomUniform:output:0=sequential/encoder_dropout5/dropout_1/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������P24
2sequential/encoder_dropout5/dropout_1/GreaterEqual�
*sequential/encoder_dropout5/dropout_1/CastCast6sequential/encoder_dropout5/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������P2,
*sequential/encoder_dropout5/dropout_1/Cast�
+sequential/encoder_dropout5/dropout_1/Mul_1Mul-sequential/encoder_dropout5/dropout_1/Mul:z:0.sequential/encoder_dropout5/dropout_1/Cast:y:0*
T0*/
_output_shapes
:���������P2-
+sequential/encoder_dropout5/dropout_1/Mul_1�
sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   2
sequential/flatten/Const_1�
sequential/flatten/Reshape_1Reshape/sequential/encoder_dropout5/dropout_1/Mul_1:z:0#sequential/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������
2
sequential/flatten/Reshape_1�
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02*
(sequential/dense/MatMul_1/ReadVariableOp�
sequential/dense/MatMul_1MatMul%sequential/flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/MatMul_1�
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)sequential/dense/BiasAdd_1/ReadVariableOp�
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/BiasAdd_1�
sequential/dense/Sigmoid_1Sigmoid#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2
sequential/dense/Sigmoid_1�
subtract/subSubsequential/dense/Sigmoid:y:0sequential/dense/Sigmoid_1:y:0*
T0*(
_output_shapes
:����������2
subtract/sub�
tf_op_layer_Abs/AbsAbssubtract/sub:z:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Abs/Abs�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMultf_op_layer_Abs/Abs:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Sigmoid�
IdentityIdentitydense_1/Sigmoid:y:0?^sequential/random_translation/stateful_uniform/StatefulUniformA^sequential/random_translation/stateful_uniform_1/StatefulUniformA^sequential/random_translation/stateful_uniform_2/StatefulUniformA^sequential/random_translation/stateful_uniform_3/StatefulUniform8^sequential/random_zoom/stateful_uniform/StatefulUniform:^sequential/random_zoom/stateful_uniform_1/StatefulUniform:^sequential/random_zoom/stateful_uniform_2/StatefulUniform:^sequential/random_zoom/stateful_uniform_3/StatefulUniform*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes|
z:�����������:�����������::::::::::::::::2�
>sequential/random_translation/stateful_uniform/StatefulUniform>sequential/random_translation/stateful_uniform/StatefulUniform2�
@sequential/random_translation/stateful_uniform_1/StatefulUniform@sequential/random_translation/stateful_uniform_1/StatefulUniform2�
@sequential/random_translation/stateful_uniform_2/StatefulUniform@sequential/random_translation/stateful_uniform_2/StatefulUniform2�
@sequential/random_translation/stateful_uniform_3/StatefulUniform@sequential/random_translation/stateful_uniform_3/StatefulUniform2r
7sequential/random_zoom/stateful_uniform/StatefulUniform7sequential/random_zoom/stateful_uniform/StatefulUniform2v
9sequential/random_zoom/stateful_uniform_1/StatefulUniform9sequential/random_zoom/stateful_uniform_1/StatefulUniform2v
9sequential/random_zoom/stateful_uniform_2/StatefulUniform9sequential/random_zoom/stateful_uniform_2/StatefulUniform2v
9sequential/random_zoom/stateful_uniform_3/StatefulUniform9sequential/random_zoom/stateful_uniform_3/StatefulUniform:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
i
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17066

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
@__inference_dense_layer_call_and_return_conditional_losses_17145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������
:::P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�
b
F__inference_random_flip_layer_call_and_return_conditional_losses_16592

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
N
2__inference_random_translation_layer_call_fn_16888

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_translation_layer_call_and_return_conditional_losses_142372
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_16946

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@:::W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
B__inference_dense_1_layer_call_and_return_conditional_losses_14947

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_functional_1_layer_call_and_return_conditional_losses_15162

inputs
inputs_1
sequential_15116
sequential_15118
sequential_15120
sequential_15122
sequential_15124
sequential_15126
sequential_15128
sequential_15130
sequential_15132
sequential_15134
sequential_15136
sequential_15138
dense_1_15156
dense_1_15158
identity��dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential/StatefulPartitionedCall_1�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_15116sequential_15118sequential_15120sequential_15122sequential_15124sequential_15126sequential_15128sequential_15130sequential_15132sequential_15134sequential_15136sequential_15138*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_147702$
"sequential/StatefulPartitionedCall�
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_15116sequential_15118sequential_15120sequential_15122sequential_15124sequential_15126sequential_15128sequential_15130sequential_15132sequential_15134sequential_15136sequential_15138*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_147702&
$sequential/StatefulPartitionedCall_1�
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_149142
subtract/PartitionedCall�
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_149282!
tf_op_layer_Abs/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_15156dense_1_15158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_149472!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
*__inference_sequential_layer_call_fn_14720
random_flip_input
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_146892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:�����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:�����������
+
_user_specified_namerandom_flip_input
�
j
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_14294

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
b
F__inference_random_zoom_layer_call_and_return_conditional_losses_16755

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
M__inference_random_translation_layer_call_and_return_conditional_losses_16876

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17061

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_16972

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������   2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������   2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�]
�
E__inference_sequential_layer_call_and_return_conditional_losses_14689

inputs
random_zoom_14641
random_translation_14644
encoder_conv1_14647
encoder_conv1_14649
encoder_conv2_14654
encoder_conv2_14656
encoder_conv3_14661
encoder_conv3_14663
encoder_conv4_14668
encoder_conv4_14670
encoder_conv5_14675
encoder_conv5_14677
dense_14683
dense_14685
identity��dense/StatefulPartitionedCall�%encoder_conv1/StatefulPartitionedCall�%encoder_conv2/StatefulPartitionedCall�%encoder_conv3/StatefulPartitionedCall�%encoder_conv4/StatefulPartitionedCall�%encoder_conv5/StatefulPartitionedCall�(encoder_dropout1/StatefulPartitionedCall�(encoder_dropout2/StatefulPartitionedCall�(encoder_dropout3/StatefulPartitionedCall�(encoder_dropout4/StatefulPartitionedCall�(encoder_dropout5/StatefulPartitionedCall�#random_flip/StatefulPartitionedCall�*random_translation/StatefulPartitionedCall�#random_zoom/StatefulPartitionedCall�
#random_flip/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_139842%
#random_flip/StatefulPartitionedCall�
#random_zoom/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_zoom_14641*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_141072%
#random_zoom/StatefulPartitionedCall�
*random_translation/StatefulPartitionedCallStatefulPartitionedCall,random_zoom/StatefulPartitionedCall:output:0random_translation_14644*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_translation_layer_call_and_return_conditional_losses_142332,
*random_translation/StatefulPartitionedCall�
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCall3random_translation/StatefulPartitionedCall:output:0encoder_conv1_14647encoder_conv1_14649*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_142652'
%encoder_conv1/StatefulPartitionedCall�
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_138992
encoder_pool1/PartitionedCall�
(encoder_dropout1/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool1/PartitionedCall:output:0$^random_flip/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_142942*
(encoder_dropout1/StatefulPartitionedCall�
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout1/StatefulPartitionedCall:output:0encoder_conv2_14654encoder_conv2_14656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_143232'
%encoder_conv2/StatefulPartitionedCall�
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_139112
encoder_pool2/PartitionedCall�
(encoder_dropout2/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool2/PartitionedCall:output:0)^encoder_dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_143522*
(encoder_dropout2/StatefulPartitionedCall�
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout2/StatefulPartitionedCall:output:0encoder_conv3_14661encoder_conv3_14663*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_143812'
%encoder_conv3/StatefulPartitionedCall�
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_139232
encoder_pool3/PartitionedCall�
(encoder_dropout3/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool3/PartitionedCall:output:0)^encoder_dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_144102*
(encoder_dropout3/StatefulPartitionedCall�
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout3/StatefulPartitionedCall:output:0encoder_conv4_14668encoder_conv4_14670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_144392'
%encoder_conv4/StatefulPartitionedCall�
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_139352
encoder_pool4/PartitionedCall�
(encoder_dropout4/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool4/PartitionedCall:output:0)^encoder_dropout3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_144682*
(encoder_dropout4/StatefulPartitionedCall�
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall1encoder_dropout4/StatefulPartitionedCall:output:0encoder_conv5_14675encoder_conv5_14677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_144972'
%encoder_conv5/StatefulPartitionedCall�
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_139472
encoder_pool5/PartitionedCall�
(encoder_dropout5/StatefulPartitionedCallStatefulPartitionedCall&encoder_pool5/PartitionedCall:output:0)^encoder_dropout4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_145262*
(encoder_dropout5/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall1encoder_dropout5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_145502
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14683dense_14685*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_145692
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall)^encoder_dropout1/StatefulPartitionedCall)^encoder_dropout2/StatefulPartitionedCall)^encoder_dropout3/StatefulPartitionedCall)^encoder_dropout4/StatefulPartitionedCall)^encoder_dropout5/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall+^random_translation/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:�����������::::::::::::::2>
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
(encoder_dropout5/StatefulPartitionedCall(encoder_dropout5/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2X
*random_translation/StatefulPartitionedCall*random_translation/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
G__inference_functional_1_layer_call_and_return_conditional_losses_15014
input_1
input_2
sequential_14968
sequential_14970
sequential_14972
sequential_14974
sequential_14976
sequential_14978
sequential_14980
sequential_14982
sequential_14984
sequential_14986
sequential_14988
sequential_14990
dense_1_15008
dense_1_15010
identity��dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential/StatefulPartitionedCall_1�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_14968sequential_14970sequential_14972sequential_14974sequential_14976sequential_14978sequential_14980sequential_14982sequential_14984sequential_14986sequential_14988sequential_14990*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_147702$
"sequential/StatefulPartitionedCall�
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_14968sequential_14970sequential_14972sequential_14974sequential_14976sequential_14978sequential_14980sequential_14982sequential_14984sequential_14986sequential_14988sequential_14990*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_147702&
$sequential/StatefulPartitionedCall_1�
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_149142
subtract/PartitionedCall�
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_149282!
tf_op_layer_Abs/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_15008dense_1_15010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_149472!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
�
G__inference_functional_1_layer_call_and_return_conditional_losses_15074

inputs
inputs_1
sequential_15022
sequential_15024
sequential_15026
sequential_15028
sequential_15030
sequential_15032
sequential_15034
sequential_15036
sequential_15038
sequential_15040
sequential_15042
sequential_15044
sequential_15046
sequential_15048
dense_1_15068
dense_1_15070
identity��dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential/StatefulPartitionedCall_1�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_15022sequential_15024sequential_15026sequential_15028sequential_15030sequential_15032sequential_15034sequential_15036sequential_15038sequential_15040sequential_15042sequential_15044sequential_15046sequential_15048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_146892$
"sequential/StatefulPartitionedCall�
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1sequential_15022sequential_15024sequential_15026sequential_15028sequential_15030sequential_15032sequential_15034sequential_15036sequential_15038sequential_15040sequential_15042sequential_15044sequential_15046sequential_15048#^sequential/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_146892&
$sequential/StatefulPartitionedCall_1�
subtract/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0-sequential/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_149142
subtract/PartitionedCall�
tf_op_layer_Abs/PartitionedCallPartitionedCall!subtract/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_149282!
tf_op_layer_Abs/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Abs/PartitionedCall:output:0dense_1_15068dense_1_15070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_149472!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes|
z:�����������:�����������::::::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_1:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_16920

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
ע
�
G__inference_functional_1_layer_call_and_return_conditional_losses_15981
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
identity��
.sequential/encoder_conv1/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.sequential/encoder_conv1/Conv2D/ReadVariableOp�
sequential/encoder_conv1/Conv2DConv2Dinputs_06sequential/encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2!
sequential/encoder_conv1/Conv2D�
/sequential/encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/encoder_conv1/BiasAdd/ReadVariableOp�
 sequential/encoder_conv1/BiasAddBiasAdd(sequential/encoder_conv1/Conv2D:output:07sequential/encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2"
 sequential/encoder_conv1/BiasAdd�
sequential/encoder_conv1/EluElu)sequential/encoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
sequential/encoder_conv1/Elu�
 sequential/encoder_pool1/MaxPoolMaxPool*sequential/encoder_conv1/Elu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool1/MaxPool�
$sequential/encoder_dropout1/IdentityIdentity)sequential/encoder_pool1/MaxPool:output:0*
T0*/
_output_shapes
:���������@@2&
$sequential/encoder_dropout1/Identity�
.sequential/encoder_conv2/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential/encoder_conv2/Conv2D/ReadVariableOp�
sequential/encoder_conv2/Conv2DConv2D-sequential/encoder_dropout1/Identity:output:06sequential/encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2!
sequential/encoder_conv2/Conv2D�
/sequential/encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/encoder_conv2/BiasAdd/ReadVariableOp�
 sequential/encoder_conv2/BiasAddBiasAdd(sequential/encoder_conv2/Conv2D:output:07sequential/encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2"
 sequential/encoder_conv2/BiasAdd�
sequential/encoder_conv2/EluElu)sequential/encoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
sequential/encoder_conv2/Elu�
 sequential/encoder_pool2/MaxPoolMaxPool*sequential/encoder_conv2/Elu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool2/MaxPool�
$sequential/encoder_dropout2/IdentityIdentity)sequential/encoder_pool2/MaxPool:output:0*
T0*/
_output_shapes
:���������   2&
$sequential/encoder_dropout2/Identity�
.sequential/encoder_conv3/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype020
.sequential/encoder_conv3/Conv2D/ReadVariableOp�
sequential/encoder_conv3/Conv2DConv2D-sequential/encoder_dropout2/Identity:output:06sequential/encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2!
sequential/encoder_conv3/Conv2D�
/sequential/encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/encoder_conv3/BiasAdd/ReadVariableOp�
 sequential/encoder_conv3/BiasAddBiasAdd(sequential/encoder_conv3/Conv2D:output:07sequential/encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02"
 sequential/encoder_conv3/BiasAdd�
sequential/encoder_conv3/EluElu)sequential/encoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:���������  02
sequential/encoder_conv3/Elu�
 sequential/encoder_pool3/MaxPoolMaxPool*sequential/encoder_conv3/Elu:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool3/MaxPool�
$sequential/encoder_dropout3/IdentityIdentity)sequential/encoder_pool3/MaxPool:output:0*
T0*/
_output_shapes
:���������02&
$sequential/encoder_dropout3/Identity�
.sequential/encoder_conv4/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype020
.sequential/encoder_conv4/Conv2D/ReadVariableOp�
sequential/encoder_conv4/Conv2DConv2D-sequential/encoder_dropout3/Identity:output:06sequential/encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2!
sequential/encoder_conv4/Conv2D�
/sequential/encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/encoder_conv4/BiasAdd/ReadVariableOp�
 sequential/encoder_conv4/BiasAddBiasAdd(sequential/encoder_conv4/Conv2D:output:07sequential/encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2"
 sequential/encoder_conv4/BiasAdd�
sequential/encoder_conv4/EluElu)sequential/encoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential/encoder_conv4/Elu�
 sequential/encoder_pool4/MaxPoolMaxPool*sequential/encoder_conv4/Elu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool4/MaxPool�
$sequential/encoder_dropout4/IdentityIdentity)sequential/encoder_pool4/MaxPool:output:0*
T0*/
_output_shapes
:���������@2&
$sequential/encoder_dropout4/Identity�
.sequential/encoder_conv5/Conv2D/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype020
.sequential/encoder_conv5/Conv2D/ReadVariableOp�
sequential/encoder_conv5/Conv2DConv2D-sequential/encoder_dropout4/Identity:output:06sequential/encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2!
sequential/encoder_conv5/Conv2D�
/sequential/encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype021
/sequential/encoder_conv5/BiasAdd/ReadVariableOp�
 sequential/encoder_conv5/BiasAddBiasAdd(sequential/encoder_conv5/Conv2D:output:07sequential/encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2"
 sequential/encoder_conv5/BiasAdd�
sequential/encoder_conv5/EluElu)sequential/encoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:���������P2
sequential/encoder_conv5/Elu�
 sequential/encoder_pool5/MaxPoolMaxPool*sequential/encoder_conv5/Elu:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2"
 sequential/encoder_pool5/MaxPool�
$sequential/encoder_dropout5/IdentityIdentity)sequential/encoder_pool5/MaxPool:output:0*
T0*/
_output_shapes
:���������P2&
$sequential/encoder_dropout5/Identity�
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
sequential/flatten/Const�
sequential/flatten/ReshapeReshape-sequential/encoder_dropout5/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:����������
2
sequential/flatten/Reshape�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/BiasAdd�
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential/dense/Sigmoid�
0sequential/encoder_conv1/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0sequential/encoder_conv1/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv1/Conv2D_1Conv2Dinputs_18sequential/encoder_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2#
!sequential/encoder_conv1/Conv2D_1�
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/encoder_conv1/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv1/BiasAdd_1BiasAdd*sequential/encoder_conv1/Conv2D_1:output:09sequential/encoder_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2$
"sequential/encoder_conv1/BiasAdd_1�
sequential/encoder_conv1/Elu_1Elu+sequential/encoder_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:�����������2 
sequential/encoder_conv1/Elu_1�
"sequential/encoder_pool1/MaxPool_1MaxPool,sequential/encoder_conv1/Elu_1:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool1/MaxPool_1�
&sequential/encoder_dropout1/Identity_1Identity+sequential/encoder_pool1/MaxPool_1:output:0*
T0*/
_output_shapes
:���������@@2(
&sequential/encoder_dropout1/Identity_1�
0sequential/encoder_conv2/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0sequential/encoder_conv2/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv2/Conv2D_1Conv2D/sequential/encoder_dropout1/Identity_1:output:08sequential/encoder_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2#
!sequential/encoder_conv2/Conv2D_1�
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential/encoder_conv2/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv2/BiasAdd_1BiasAdd*sequential/encoder_conv2/Conv2D_1:output:09sequential/encoder_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2$
"sequential/encoder_conv2/BiasAdd_1�
sequential/encoder_conv2/Elu_1Elu+sequential/encoder_conv2/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������@@ 2 
sequential/encoder_conv2/Elu_1�
"sequential/encoder_pool2/MaxPool_1MaxPool,sequential/encoder_conv2/Elu_1:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool2/MaxPool_1�
&sequential/encoder_dropout2/Identity_1Identity+sequential/encoder_pool2/MaxPool_1:output:0*
T0*/
_output_shapes
:���������   2(
&sequential/encoder_dropout2/Identity_1�
0sequential/encoder_conv3/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype022
0sequential/encoder_conv3/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv3/Conv2D_1Conv2D/sequential/encoder_dropout2/Identity_1:output:08sequential/encoder_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2#
!sequential/encoder_conv3/Conv2D_1�
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1sequential/encoder_conv3/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv3/BiasAdd_1BiasAdd*sequential/encoder_conv3/Conv2D_1:output:09sequential/encoder_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02$
"sequential/encoder_conv3/BiasAdd_1�
sequential/encoder_conv3/Elu_1Elu+sequential/encoder_conv3/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������  02 
sequential/encoder_conv3/Elu_1�
"sequential/encoder_pool3/MaxPool_1MaxPool,sequential/encoder_conv3/Elu_1:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool3/MaxPool_1�
&sequential/encoder_dropout3/Identity_1Identity+sequential/encoder_pool3/MaxPool_1:output:0*
T0*/
_output_shapes
:���������02(
&sequential/encoder_dropout3/Identity_1�
0sequential/encoder_conv4/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype022
0sequential/encoder_conv4/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv4/Conv2D_1Conv2D/sequential/encoder_dropout3/Identity_1:output:08sequential/encoder_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!sequential/encoder_conv4/Conv2D_1�
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential/encoder_conv4/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv4/BiasAdd_1BiasAdd*sequential/encoder_conv4/Conv2D_1:output:09sequential/encoder_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2$
"sequential/encoder_conv4/BiasAdd_1�
sequential/encoder_conv4/Elu_1Elu+sequential/encoder_conv4/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������@2 
sequential/encoder_conv4/Elu_1�
"sequential/encoder_pool4/MaxPool_1MaxPool,sequential/encoder_conv4/Elu_1:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool4/MaxPool_1�
&sequential/encoder_dropout4/Identity_1Identity+sequential/encoder_pool4/MaxPool_1:output:0*
T0*/
_output_shapes
:���������@2(
&sequential/encoder_dropout4/Identity_1�
0sequential/encoder_conv5/Conv2D_1/ReadVariableOpReadVariableOp7sequential_encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype022
0sequential/encoder_conv5/Conv2D_1/ReadVariableOp�
!sequential/encoder_conv5/Conv2D_1Conv2D/sequential/encoder_dropout4/Identity_1:output:08sequential/encoder_conv5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2#
!sequential/encoder_conv5/Conv2D_1�
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOpReadVariableOp8sequential_encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype023
1sequential/encoder_conv5/BiasAdd_1/ReadVariableOp�
"sequential/encoder_conv5/BiasAdd_1BiasAdd*sequential/encoder_conv5/Conv2D_1:output:09sequential/encoder_conv5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2$
"sequential/encoder_conv5/BiasAdd_1�
sequential/encoder_conv5/Elu_1Elu+sequential/encoder_conv5/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������P2 
sequential/encoder_conv5/Elu_1�
"sequential/encoder_pool5/MaxPool_1MaxPool,sequential/encoder_conv5/Elu_1:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2$
"sequential/encoder_pool5/MaxPool_1�
&sequential/encoder_dropout5/Identity_1Identity+sequential/encoder_pool5/MaxPool_1:output:0*
T0*/
_output_shapes
:���������P2(
&sequential/encoder_dropout5/Identity_1�
sequential/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   2
sequential/flatten/Const_1�
sequential/flatten/Reshape_1Reshape/sequential/encoder_dropout5/Identity_1:output:0#sequential/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������
2
sequential/flatten/Reshape_1�
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02*
(sequential/dense/MatMul_1/ReadVariableOp�
sequential/dense/MatMul_1MatMul%sequential/flatten/Reshape_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/MatMul_1�
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)sequential/dense/BiasAdd_1/ReadVariableOp�
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential/dense/BiasAdd_1�
sequential/dense/Sigmoid_1Sigmoid#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:����������2
sequential/dense/Sigmoid_1�
subtract/subSubsequential/dense/Sigmoid:y:0sequential/dense/Sigmoid_1:y:0*
T0*(
_output_shapes
:����������2
subtract/sub�
tf_op_layer_Abs/AbsAbssubtract/sub:z:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Abs/Abs�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMultf_op_layer_Abs/Abs:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Sigmoidg
IdentityIdentitydense_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������:::::::::::::::[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�	
�
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_17040

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������0:::W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
,__inference_functional_1_layer_call_fn_15193
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
identity��StatefulPartitionedCall�
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
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_151622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapest
r:�����������:�����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
d
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_13899

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�`
�
F__inference_random_flip_layer_call_and_return_conditional_losses_16588

inputs
identity��9random_flip_left_right/assert_greater_equal/Assert/Assert�@random_flip_left_right/assert_positive/assert_less/Assert/Assertr
random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2
random_flip_left_right/Shape�
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*random_flip_left_right/strided_slice/stack�
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice/stack_1�
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2�
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$random_flip_left_right/strided_slice�
,random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,random_flip_left_right/assert_positive/Const�
7random_flip_left_right/assert_positive/assert_less/LessLess5random_flip_left_right/assert_positive/Const:output:0-random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:29
7random_flip_left_right/assert_positive/assert_less/Less�
8random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_left_right/assert_positive/assert_less/Const�
6random_flip_left_right/assert_positive/assert_less/AllAll;random_flip_left_right/assert_positive/assert_less/Less:z:0Arandom_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 28
6random_flip_left_right/assert_positive/assert_less/All�
?random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2A
?random_flip_left_right/assert_positive/assert_less/Assert/Const�
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2I
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0�
@random_flip_left_right/assert_positive/assert_less/Assert/AssertAssert?random_flip_left_right/assert_positive/assert_less/All:output:0Prandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2B
@random_flip_left_right/assert_positive/assert_less/Assert/Assert|
random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2
random_flip_left_right/Rank�
-random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_flip_left_right/assert_greater_equal/y�
8random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual$random_flip_left_right/Rank:output:06random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2:
8random_flip_left_right/assert_greater_equal/GreaterEqual�
1random_flip_left_right/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 23
1random_flip_left_right/assert_greater_equal/Const�
/random_flip_left_right/assert_greater_equal/AllAll<random_flip_left_right/assert_greater_equal/GreaterEqual:z:0:random_flip_left_right/assert_greater_equal/Const:output:0*
_output_shapes
: 21
/random_flip_left_right/assert_greater_equal/All�
8random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2:
8random_flip_left_right/assert_greater_equal/Assert/Const�
:random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2<
:random_flip_left_right/assert_greater_equal/Assert/Const_1�
:random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_2�
:random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_3�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2�
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4�
9random_flip_left_right/assert_greater_equal/Assert/AssertAssert8random_flip_left_right/assert_greater_equal/All:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0$random_flip_left_right/Rank:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:06random_flip_left_right/assert_greater_equal/y:output:0A^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2;
9random_flip_left_right/assert_greater_equal/Assert/Assert�
)random_flip_left_right/control_dependencyIdentityinputs:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4������������������������������������2+
)random_flip_left_right/control_dependency�
random_flip_left_right/Shape_1Shape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2 
random_flip_left_right/Shape_1�
,random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice_1/stack�
.random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_1�
.random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_2�
&random_flip_left_right/strided_slice_1StridedSlice'random_flip_left_right/Shape_1:output:05random_flip_left_right/strided_slice_1/stack:output:07random_flip_left_right/strided_slice_1/stack_1:output:07random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&random_flip_left_right/strided_slice_1�
+random_flip_left_right/random_uniform/shapePack/random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape�
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min�
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)random_flip_left_right/random_uniform/max�
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype025
3random_flip_left_right/random_uniform/RandomUniform�
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:���������2+
)random_flip_left_right/random_uniform/Mul�
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1�
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2�
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3�
$random_flip_left_right/Reshape/shapePack/random_flip_left_right/strided_slice_1:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shape�
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2 
random_flip_left_right/Reshape�
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/Round�
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axis�
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4������������������������������������2"
 random_flip_left_right/ReverseV2�
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
random_flip_left_right/mul�
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_flip_left_right/sub/x�
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:���������2
random_flip_left_right/sub�
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
random_flip_left_right/mul_1�
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2
random_flip_left_right/add�
IdentityIdentityrandom_flip_left_right/add:z:0:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������2v
9random_flip_left_right/assert_greater_equal/Assert/Assert9random_flip_left_right/assert_greater_equal/Assert/Assert2�
@random_flip_left_right/assert_positive/assert_less/Assert/Assert@random_flip_left_right/assert_positive/assert_less/Assert/Assert:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_14323

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@@:::W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�N
�
E__inference_sequential_layer_call_and_return_conditional_losses_14634
random_flip_input
encoder_conv1_14592
encoder_conv1_14594
encoder_conv2_14599
encoder_conv2_14601
encoder_conv3_14606
encoder_conv3_14608
encoder_conv4_14613
encoder_conv4_14615
encoder_conv5_14620
encoder_conv5_14622
dense_14628
dense_14630
identity��dense/StatefulPartitionedCall�%encoder_conv1/StatefulPartitionedCall�%encoder_conv2/StatefulPartitionedCall�%encoder_conv3/StatefulPartitionedCall�%encoder_conv4/StatefulPartitionedCall�%encoder_conv5/StatefulPartitionedCall�
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_139882
random_flip/PartitionedCall�
random_zoom/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_141112
random_zoom/PartitionedCall�
"random_translation/PartitionedCallPartitionedCall$random_zoom/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_translation_layer_call_and_return_conditional_losses_142372$
"random_translation/PartitionedCall�
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCall+random_translation/PartitionedCall:output:0encoder_conv1_14592encoder_conv1_14594*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_142652'
%encoder_conv1/StatefulPartitionedCall�
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_138992
encoder_pool1/PartitionedCall�
 encoder_dropout1/PartitionedCallPartitionedCall&encoder_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_142992"
 encoder_dropout1/PartitionedCall�
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout1/PartitionedCall:output:0encoder_conv2_14599encoder_conv2_14601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_143232'
%encoder_conv2/StatefulPartitionedCall�
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_139112
encoder_pool2/PartitionedCall�
 encoder_dropout2/PartitionedCallPartitionedCall&encoder_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_143572"
 encoder_dropout2/PartitionedCall�
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout2/PartitionedCall:output:0encoder_conv3_14606encoder_conv3_14608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_143812'
%encoder_conv3/StatefulPartitionedCall�
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_139232
encoder_pool3/PartitionedCall�
 encoder_dropout3/PartitionedCallPartitionedCall&encoder_pool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_144152"
 encoder_dropout3/PartitionedCall�
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout3/PartitionedCall:output:0encoder_conv4_14613encoder_conv4_14615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_144392'
%encoder_conv4/StatefulPartitionedCall�
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_139352
encoder_pool4/PartitionedCall�
 encoder_dropout4/PartitionedCallPartitionedCall&encoder_pool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_144732"
 encoder_dropout4/PartitionedCall�
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout4/PartitionedCall:output:0encoder_conv5_14620encoder_conv5_14622*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_144972'
%encoder_conv5/StatefulPartitionedCall�
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_139472
encoder_pool5/PartitionedCall�
 encoder_dropout5/PartitionedCallPartitionedCall&encoder_pool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_145312"
 encoder_dropout5/PartitionedCall�
flatten/PartitionedCallPartitionedCall)encoder_dropout5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_145502
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14628dense_14630*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_145692
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%encoder_conv1/StatefulPartitionedCall%encoder_conv1/StatefulPartitionedCall2N
%encoder_conv2/StatefulPartitionedCall%encoder_conv2/StatefulPartitionedCall2N
%encoder_conv3/StatefulPartitionedCall%encoder_conv3/StatefulPartitionedCall2N
%encoder_conv4/StatefulPartitionedCall%encoder_conv4/StatefulPartitionedCall2N
%encoder_conv5/StatefulPartitionedCall%encoder_conv5/StatefulPartitionedCall:d `
1
_output_shapes
:�����������
+
_user_specified_namerandom_flip_input
�
I
-__inference_encoder_pool5_layer_call_fn_13953

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_139472
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
˴
�
!__inference__traced_restore_17455
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
assignvariableop_19_variable"
assignvariableop_20_variable_1"
assignvariableop_21_variable_2
assignvariableop_22_total
assignvariableop_23_count&
"assignvariableop_24_true_positives&
"assignvariableop_25_true_negatives'
#assignvariableop_26_false_positives'
#assignvariableop_27_false_negatives2
.assignvariableop_28_rmsprop_dense_1_kernel_rms0
,assignvariableop_29_rmsprop_dense_1_bias_rmsC
?assignvariableop_30_rmsprop_sequential_encoder_conv1_kernel_rmsA
=assignvariableop_31_rmsprop_sequential_encoder_conv1_bias_rmsC
?assignvariableop_32_rmsprop_sequential_encoder_conv2_kernel_rmsA
=assignvariableop_33_rmsprop_sequential_encoder_conv2_bias_rmsC
?assignvariableop_34_rmsprop_sequential_encoder_conv3_kernel_rmsA
=assignvariableop_35_rmsprop_sequential_encoder_conv3_bias_rmsC
?assignvariableop_36_rmsprop_sequential_encoder_conv4_kernel_rmsA
=assignvariableop_37_rmsprop_sequential_encoder_conv4_bias_rmsC
?assignvariableop_38_rmsprop_sequential_encoder_conv5_kernel_rmsA
=assignvariableop_39_rmsprop_sequential_encoder_conv5_bias_rms;
7assignvariableop_40_rmsprop_sequential_dense_kernel_rms9
5assignvariableop_41_rmsprop_sequential_dense_bias_rms
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+				2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_momentumIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_rhoIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp2assignvariableop_7_sequential_encoder_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_sequential_encoder_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_sequential_encoder_conv2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_sequential_encoder_conv2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp3assignvariableop_11_sequential_encoder_conv3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp1assignvariableop_12_sequential_encoder_conv3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp3assignvariableop_13_sequential_encoder_conv4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_sequential_encoder_conv4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp3assignvariableop_15_sequential_encoder_conv5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_sequential_encoder_conv5_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_sequential_dense_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_sequential_dense_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variableIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_true_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_true_negativesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp#assignvariableop_26_false_positivesIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_false_negativesIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_rmsprop_dense_1_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp,assignvariableop_29_rmsprop_dense_1_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp?assignvariableop_30_rmsprop_sequential_encoder_conv1_kernel_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp=assignvariableop_31_rmsprop_sequential_encoder_conv1_bias_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp?assignvariableop_32_rmsprop_sequential_encoder_conv2_kernel_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp=assignvariableop_33_rmsprop_sequential_encoder_conv2_bias_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp?assignvariableop_34_rmsprop_sequential_encoder_conv3_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp=assignvariableop_35_rmsprop_sequential_encoder_conv3_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp?assignvariableop_36_rmsprop_sequential_encoder_conv4_kernel_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp=assignvariableop_37_rmsprop_sequential_encoder_conv4_bias_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp?assignvariableop_38_rmsprop_sequential_encoder_conv5_kernel_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp=assignvariableop_39_rmsprop_sequential_encoder_conv5_bias_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp7assignvariableop_40_rmsprop_sequential_dense_kernel_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_rmsprop_sequential_dense_bias_rmsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42�
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
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
�
d
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_13947

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17014

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������02
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������02
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_14352

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������   2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������   *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������   2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������   2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������   2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
i
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_14299

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@@:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
i
0__inference_encoder_dropout4_layer_call_fn_17071

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_144682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�N
�
E__inference_sequential_layer_call_and_return_conditional_losses_14770

inputs
encoder_conv1_14728
encoder_conv1_14730
encoder_conv2_14735
encoder_conv2_14737
encoder_conv3_14742
encoder_conv3_14744
encoder_conv4_14749
encoder_conv4_14751
encoder_conv5_14756
encoder_conv5_14758
dense_14764
dense_14766
identity��dense/StatefulPartitionedCall�%encoder_conv1/StatefulPartitionedCall�%encoder_conv2/StatefulPartitionedCall�%encoder_conv3/StatefulPartitionedCall�%encoder_conv4/StatefulPartitionedCall�%encoder_conv5/StatefulPartitionedCall�
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_139882
random_flip/PartitionedCall�
random_zoom/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_141112
random_zoom/PartitionedCall�
"random_translation/PartitionedCallPartitionedCall$random_zoom/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_translation_layer_call_and_return_conditional_losses_142372$
"random_translation/PartitionedCall�
%encoder_conv1/StatefulPartitionedCallStatefulPartitionedCall+random_translation/PartitionedCall:output:0encoder_conv1_14728encoder_conv1_14730*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_142652'
%encoder_conv1/StatefulPartitionedCall�
encoder_pool1/PartitionedCallPartitionedCall.encoder_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_138992
encoder_pool1/PartitionedCall�
 encoder_dropout1/PartitionedCallPartitionedCall&encoder_pool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_142992"
 encoder_dropout1/PartitionedCall�
%encoder_conv2/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout1/PartitionedCall:output:0encoder_conv2_14735encoder_conv2_14737*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_143232'
%encoder_conv2/StatefulPartitionedCall�
encoder_pool2/PartitionedCallPartitionedCall.encoder_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_139112
encoder_pool2/PartitionedCall�
 encoder_dropout2/PartitionedCallPartitionedCall&encoder_pool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_143572"
 encoder_dropout2/PartitionedCall�
%encoder_conv3/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout2/PartitionedCall:output:0encoder_conv3_14742encoder_conv3_14744*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_143812'
%encoder_conv3/StatefulPartitionedCall�
encoder_pool3/PartitionedCallPartitionedCall.encoder_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_139232
encoder_pool3/PartitionedCall�
 encoder_dropout3/PartitionedCallPartitionedCall&encoder_pool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_144152"
 encoder_dropout3/PartitionedCall�
%encoder_conv4/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout3/PartitionedCall:output:0encoder_conv4_14749encoder_conv4_14751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_144392'
%encoder_conv4/StatefulPartitionedCall�
encoder_pool4/PartitionedCallPartitionedCall.encoder_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_139352
encoder_pool4/PartitionedCall�
 encoder_dropout4/PartitionedCallPartitionedCall&encoder_pool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_144732"
 encoder_dropout4/PartitionedCall�
%encoder_conv5/StatefulPartitionedCallStatefulPartitionedCall)encoder_dropout4/PartitionedCall:output:0encoder_conv5_14756encoder_conv5_14758*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_144972'
%encoder_conv5/StatefulPartitionedCall�
encoder_pool5/PartitionedCallPartitionedCall.encoder_conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_139472
encoder_pool5/PartitionedCall�
 encoder_dropout5/PartitionedCallPartitionedCall&encoder_pool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_145312"
 encoder_dropout5/PartitionedCall�
flatten/PartitionedCallPartitionedCall)encoder_dropout5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_145502
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14764dense_14766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_145692
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall&^encoder_conv1/StatefulPartitionedCall&^encoder_conv2/StatefulPartitionedCall&^encoder_conv3/StatefulPartitionedCall&^encoder_conv4/StatefulPartitionedCall&^encoder_conv5/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%encoder_conv1/StatefulPartitionedCall%encoder_conv1/StatefulPartitionedCall2N
%encoder_conv2/StatefulPartitionedCall%encoder_conv2/StatefulPartitionedCall2N
%encoder_conv3/StatefulPartitionedCall%encoder_conv3/StatefulPartitionedCall2N
%encoder_conv4/StatefulPartitionedCall%encoder_conv4/StatefulPartitionedCall2N
%encoder_conv5/StatefulPartitionedCall%encoder_conv5/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
0__inference_encoder_dropout1_layer_call_fn_16930

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_142942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
G
+__inference_random_flip_layer_call_fn_16644

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_139882
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
F__inference_random_zoom_layer_call_and_return_conditional_losses_14107

inputs-
)stateful_uniform_statefuluniform_resource
identity�� stateful_uniform/StatefulUniform�"stateful_uniform_1/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1�
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
stateful_uniform/max�
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm�
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype02"
 stateful_uniform/StatefulUniform�
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub�
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform/mul�
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniformz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1�
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform_1/shapeu
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
stateful_uniform_1/max�
,stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2.
,stateful_uniform_1/StatefulUniform/algorithm�
"stateful_uniform_1/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource5stateful_uniform_1/StatefulUniform/algorithm:output:0!stateful_uniform_1/shape:output:0!^stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype02$
"stateful_uniform_1/StatefulUniform�
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub�
stateful_uniform_1/mulMul+stateful_uniform_1/StatefulUniform:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1/mul�
stateful_uniform_1Addstateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2
stateful_uniform_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2stateful_uniform_1:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shape�
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stack�
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1�
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2�
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zoom_matrix/strided_slicek
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub/yr
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/subs
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv/y�
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv�
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stack�
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1�
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2�
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_1o
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub_1/x�
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/sub_1�
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:���������2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub_2/yv
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/sub_2w
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv_1/y�
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1�
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stack�
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1�
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2�
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_2o
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
zoom_matrix/sub_3/x�
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/sub_3�
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:���������2
zoom_matrix/mul_1�
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stack�
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1�
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2�
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_3t
zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/mul/y�
zoom_matrix/zeros/mulMul"zoom_matrix/strided_slice:output:0 zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros/mulw
zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zoom_matrix/zeros/Less/y�
zoom_matrix/zeros/LessLesszoom_matrix/zeros/mul:z:0!zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros/Lessz
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/packed/1�
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros/packedw
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros/Const�
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/zerosx
zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/mul/y�
zoom_matrix/zeros_1/mulMul"zoom_matrix/strided_slice:output:0"zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_1/mul{
zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zoom_matrix/zeros_1/Less/y�
zoom_matrix/zeros_1/LessLesszoom_matrix/zeros_1/mul:z:0#zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_1/Less~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1�
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_1/packed{
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_1/Const�
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/zeros_1�
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stack�
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1�
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2�
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_4x
zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/mul/y�
zoom_matrix/zeros_2/mulMul"zoom_matrix/strided_slice:output:0"zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_2/mul{
zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zoom_matrix/zeros_2/Less/y�
zoom_matrix/zeros_2/LessLesszoom_matrix/zeros_2/mul:z:0#zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/zeros_2/Less~
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/packed/1�
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_2/packed{
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_2/Const�
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axis�
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape�
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack�
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1�
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice�
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputszoom_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2�
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform#^stateful_uniform_1/StatefulUniform*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform2H
"stateful_uniform_1/StatefulUniform"stateful_uniform_1/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
I
-__inference_encoder_pool3_layer_call_fn_13929

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_139232
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_random_translation_layer_call_and_return_conditional_losses_14237

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
E__inference_sequential_layer_call_and_return_conditional_losses_16374

inputs9
5random_zoom_stateful_uniform_statefuluniform_resource@
<random_translation_stateful_uniform_statefuluniform_resource0
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
identity��3random_translation/stateful_uniform/StatefulUniform�5random_translation/stateful_uniform_1/StatefulUniform�,random_zoom/stateful_uniform/StatefulUniform�.random_zoom/stateful_uniform_1/StatefulUniform�
5random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:�����������27
5random_flip/random_flip_left_right/control_dependency�
(random_flip/random_flip_left_right/ShapeShape>random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip/random_flip_left_right/Shape�
6random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip/random_flip_left_right/strided_slice/stack�
8random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_1�
8random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_2�
0random_flip/random_flip_left_right/strided_sliceStridedSlice1random_flip/random_flip_left_right/Shape:output:0?random_flip/random_flip_left_right/strided_slice/stack:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_1:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip/random_flip_left_right/strided_slice�
7random_flip/random_flip_left_right/random_uniform/shapePack9random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip/random_flip_left_right/random_uniform/shape�
5random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip/random_flip_left_right/random_uniform/min�
5random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?27
5random_flip/random_flip_left_right/random_uniform/max�
?random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform@random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:���������*
dtype02A
?random_flip/random_flip_left_right/random_uniform/RandomUniform�
5random_flip/random_flip_left_right/random_uniform/MulMulHrandom_flip/random_flip_left_right/random_uniform/RandomUniform:output:0>random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:���������27
5random_flip/random_flip_left_right/random_uniform/Mul�
2random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/1�
2random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/2�
2random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/3�
0random_flip/random_flip_left_right/Reshape/shapePack9random_flip/random_flip_left_right/strided_slice:output:0;random_flip/random_flip_left_right/Reshape/shape/1:output:0;random_flip/random_flip_left_right/Reshape/shape/2:output:0;random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip/random_flip_left_right/Reshape/shape�
*random_flip/random_flip_left_right/ReshapeReshape9random_flip/random_flip_left_right/random_uniform/Mul:z:09random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2,
*random_flip/random_flip_left_right/Reshape�
(random_flip/random_flip_left_right/RoundRound3random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������2*
(random_flip/random_flip_left_right/Round�
1random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip/random_flip_left_right/ReverseV2/axis�
,random_flip/random_flip_left_right/ReverseV2	ReverseV2>random_flip/random_flip_left_right/control_dependency:output:0:random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:�����������2.
,random_flip/random_flip_left_right/ReverseV2�
&random_flip/random_flip_left_right/mulMul,random_flip/random_flip_left_right/Round:y:05random_flip/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:�����������2(
&random_flip/random_flip_left_right/mul�
(random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2*
(random_flip/random_flip_left_right/sub/x�
&random_flip/random_flip_left_right/subSub1random_flip/random_flip_left_right/sub/x:output:0,random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:���������2(
&random_flip/random_flip_left_right/sub�
(random_flip/random_flip_left_right/mul_1Mul*random_flip/random_flip_left_right/sub:z:0>random_flip/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:�����������2*
(random_flip/random_flip_left_right/mul_1�
&random_flip/random_flip_left_right/addAddV2*random_flip/random_flip_left_right/mul:z:0,random_flip/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:�����������2(
&random_flip/random_flip_left_right/add�
random_zoom/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_zoom/Shape�
random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
random_zoom/strided_slice/stack�
!random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_zoom/strided_slice/stack_1�
!random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!random_zoom/strided_slice/stack_2�
random_zoom/strided_sliceStridedSlicerandom_zoom/Shape:output:0(random_zoom/strided_slice/stack:output:0*random_zoom/strided_slice/stack_1:output:0*random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom/strided_slice�
!random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_zoom/strided_slice_1/stack�
#random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom/strided_slice_1/stack_1�
#random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom/strided_slice_1/stack_2�
random_zoom/strided_slice_1StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_1/stack:output:0,random_zoom/strided_slice_1/stack_1:output:0,random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom/strided_slice_1�
random_zoom/CastCast$random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom/Cast�
!random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!random_zoom/strided_slice_2/stack�
#random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom/strided_slice_2/stack_1�
#random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom/strided_slice_2/stack_2�
random_zoom/strided_slice_2StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_2/stack:output:0,random_zoom/strided_slice_2/stack_1:output:0,random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom/strided_slice_2�
random_zoom/Cast_1Cast$random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom/Cast_1�
$random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$random_zoom/stateful_uniform/shape/1�
"random_zoom/stateful_uniform/shapePack"random_zoom/strided_slice:output:0-random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"random_zoom/stateful_uniform/shape�
 random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2"
 random_zoom/stateful_uniform/min�
 random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2"
 random_zoom/stateful_uniform/max�
6random_zoom/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R28
6random_zoom/stateful_uniform/StatefulUniform/algorithm�
,random_zoom/stateful_uniform/StatefulUniformStatefulUniform5random_zoom_stateful_uniform_statefuluniform_resource?random_zoom/stateful_uniform/StatefulUniform/algorithm:output:0+random_zoom/stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype02.
,random_zoom/stateful_uniform/StatefulUniform�
 random_zoom/stateful_uniform/subSub)random_zoom/stateful_uniform/max:output:0)random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2"
 random_zoom/stateful_uniform/sub�
 random_zoom/stateful_uniform/mulMul5random_zoom/stateful_uniform/StatefulUniform:output:0$random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2"
 random_zoom/stateful_uniform/mul�
random_zoom/stateful_uniformAdd$random_zoom/stateful_uniform/mul:z:0)random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2
random_zoom/stateful_uniform�
&random_zoom/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom/stateful_uniform_1/shape/1�
$random_zoom/stateful_uniform_1/shapePack"random_zoom/strided_slice:output:0/random_zoom/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom/stateful_uniform_1/shape�
"random_zoom/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?2$
"random_zoom/stateful_uniform_1/min�
"random_zoom/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?2$
"random_zoom/stateful_uniform_1/max�
8random_zoom/stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2:
8random_zoom/stateful_uniform_1/StatefulUniform/algorithm�
.random_zoom/stateful_uniform_1/StatefulUniformStatefulUniform5random_zoom_stateful_uniform_statefuluniform_resourceArandom_zoom/stateful_uniform_1/StatefulUniform/algorithm:output:0-random_zoom/stateful_uniform_1/shape:output:0-^random_zoom/stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype020
.random_zoom/stateful_uniform_1/StatefulUniform�
"random_zoom/stateful_uniform_1/subSub+random_zoom/stateful_uniform_1/max:output:0+random_zoom/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2$
"random_zoom/stateful_uniform_1/sub�
"random_zoom/stateful_uniform_1/mulMul7random_zoom/stateful_uniform_1/StatefulUniform:output:0&random_zoom/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2$
"random_zoom/stateful_uniform_1/mul�
random_zoom/stateful_uniform_1Add&random_zoom/stateful_uniform_1/mul:z:0+random_zoom/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2 
random_zoom/stateful_uniform_1t
random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom/concat/axis�
random_zoom/concatConcatV2"random_zoom/stateful_uniform_1:z:0 random_zoom/stateful_uniform:z:0 random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
random_zoom/concat�
random_zoom/zoom_matrix/ShapeShaperandom_zoom/concat:output:0*
T0*
_output_shapes
:2
random_zoom/zoom_matrix/Shape�
+random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+random_zoom/zoom_matrix/strided_slice/stack�
-random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom/zoom_matrix/strided_slice/stack_1�
-random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom/zoom_matrix/strided_slice/stack_2�
%random_zoom/zoom_matrix/strided_sliceStridedSlice&random_zoom/zoom_matrix/Shape:output:04random_zoom/zoom_matrix/strided_slice/stack:output:06random_zoom/zoom_matrix/strided_slice/stack_1:output:06random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%random_zoom/zoom_matrix/strided_slice�
random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
random_zoom/zoom_matrix/sub/y�
random_zoom/zoom_matrix/subSubrandom_zoom/Cast_1:y:0&random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
random_zoom/zoom_matrix/sub�
!random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!random_zoom/zoom_matrix/truediv/y�
random_zoom/zoom_matrix/truedivRealDivrandom_zoom/zoom_matrix/sub:z:0*random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2!
random_zoom/zoom_matrix/truediv�
-random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2/
-random_zoom/zoom_matrix/strided_slice_1/stack�
/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom/zoom_matrix/strided_slice_1/stack_1�
/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/random_zoom/zoom_matrix/strided_slice_1/stack_2�
'random_zoom/zoom_matrix/strided_slice_1StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_1/stack:output:08random_zoom/zoom_matrix/strided_slice_1/stack_1:output:08random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2)
'random_zoom/zoom_matrix/strided_slice_1�
random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
random_zoom/zoom_matrix/sub_1/x�
random_zoom/zoom_matrix/sub_1Sub(random_zoom/zoom_matrix/sub_1/x:output:00random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2
random_zoom/zoom_matrix/sub_1�
random_zoom/zoom_matrix/mulMul#random_zoom/zoom_matrix/truediv:z:0!random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:���������2
random_zoom/zoom_matrix/mul�
random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
random_zoom/zoom_matrix/sub_2/y�
random_zoom/zoom_matrix/sub_2Subrandom_zoom/Cast:y:0(random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
random_zoom/zoom_matrix/sub_2�
#random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#random_zoom/zoom_matrix/truediv_1/y�
!random_zoom/zoom_matrix/truediv_1RealDiv!random_zoom/zoom_matrix/sub_2:z:0,random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom/zoom_matrix/truediv_1�
-random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-random_zoom/zoom_matrix/strided_slice_2/stack�
/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom/zoom_matrix/strided_slice_2/stack_1�
/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/random_zoom/zoom_matrix/strided_slice_2/stack_2�
'random_zoom/zoom_matrix/strided_slice_2StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_2/stack:output:08random_zoom/zoom_matrix/strided_slice_2/stack_1:output:08random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2)
'random_zoom/zoom_matrix/strided_slice_2�
random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
random_zoom/zoom_matrix/sub_3/x�
random_zoom/zoom_matrix/sub_3Sub(random_zoom/zoom_matrix/sub_3/x:output:00random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
random_zoom/zoom_matrix/sub_3�
random_zoom/zoom_matrix/mul_1Mul%random_zoom/zoom_matrix/truediv_1:z:0!random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:���������2
random_zoom/zoom_matrix/mul_1�
-random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2/
-random_zoom/zoom_matrix/strided_slice_3/stack�
/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom/zoom_matrix/strided_slice_3/stack_1�
/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/random_zoom/zoom_matrix/strided_slice_3/stack_2�
'random_zoom/zoom_matrix/strided_slice_3StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_3/stack:output:08random_zoom/zoom_matrix/strided_slice_3/stack_1:output:08random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2)
'random_zoom/zoom_matrix/strided_slice_3�
#random_zoom/zoom_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#random_zoom/zoom_matrix/zeros/mul/y�
!random_zoom/zoom_matrix/zeros/mulMul.random_zoom/zoom_matrix/strided_slice:output:0,random_zoom/zoom_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom/zoom_matrix/zeros/mul�
$random_zoom/zoom_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$random_zoom/zoom_matrix/zeros/Less/y�
"random_zoom/zoom_matrix/zeros/LessLess%random_zoom/zoom_matrix/zeros/mul:z:0-random_zoom/zoom_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"random_zoom/zoom_matrix/zeros/Less�
&random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom/zoom_matrix/zeros/packed/1�
$random_zoom/zoom_matrix/zeros/packedPack.random_zoom/zoom_matrix/strided_slice:output:0/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom/zoom_matrix/zeros/packed�
#random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#random_zoom/zoom_matrix/zeros/Const�
random_zoom/zoom_matrix/zerosFill-random_zoom/zoom_matrix/zeros/packed:output:0,random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
random_zoom/zoom_matrix/zeros�
%random_zoom/zoom_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom/zoom_matrix/zeros_1/mul/y�
#random_zoom/zoom_matrix/zeros_1/mulMul.random_zoom/zoom_matrix/strided_slice:output:0.random_zoom/zoom_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom/zoom_matrix/zeros_1/mul�
&random_zoom/zoom_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2(
&random_zoom/zoom_matrix/zeros_1/Less/y�
$random_zoom/zoom_matrix/zeros_1/LessLess'random_zoom/zoom_matrix/zeros_1/mul:z:0/random_zoom/zoom_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom/zoom_matrix/zeros_1/Less�
(random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom/zoom_matrix/zeros_1/packed/1�
&random_zoom/zoom_matrix/zeros_1/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom/zoom_matrix/zeros_1/packed�
%random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom/zoom_matrix/zeros_1/Const�
random_zoom/zoom_matrix/zeros_1Fill/random_zoom/zoom_matrix/zeros_1/packed:output:0.random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2!
random_zoom/zoom_matrix/zeros_1�
-random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-random_zoom/zoom_matrix/strided_slice_4/stack�
/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom/zoom_matrix/strided_slice_4/stack_1�
/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/random_zoom/zoom_matrix/strided_slice_4/stack_2�
'random_zoom/zoom_matrix/strided_slice_4StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_4/stack:output:08random_zoom/zoom_matrix/strided_slice_4/stack_1:output:08random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2)
'random_zoom/zoom_matrix/strided_slice_4�
%random_zoom/zoom_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom/zoom_matrix/zeros_2/mul/y�
#random_zoom/zoom_matrix/zeros_2/mulMul.random_zoom/zoom_matrix/strided_slice:output:0.random_zoom/zoom_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom/zoom_matrix/zeros_2/mul�
&random_zoom/zoom_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2(
&random_zoom/zoom_matrix/zeros_2/Less/y�
$random_zoom/zoom_matrix/zeros_2/LessLess'random_zoom/zoom_matrix/zeros_2/mul:z:0/random_zoom/zoom_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom/zoom_matrix/zeros_2/Less�
(random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom/zoom_matrix/zeros_2/packed/1�
&random_zoom/zoom_matrix/zeros_2/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom/zoom_matrix/zeros_2/packed�
%random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom/zoom_matrix/zeros_2/Const�
random_zoom/zoom_matrix/zeros_2Fill/random_zoom/zoom_matrix/zeros_2/packed:output:0.random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2!
random_zoom/zoom_matrix/zeros_2�
#random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#random_zoom/zoom_matrix/concat/axis�
random_zoom/zoom_matrix/concatConcatV20random_zoom/zoom_matrix/strided_slice_3:output:0&random_zoom/zoom_matrix/zeros:output:0random_zoom/zoom_matrix/mul:z:0(random_zoom/zoom_matrix/zeros_1:output:00random_zoom/zoom_matrix/strided_slice_4:output:0!random_zoom/zoom_matrix/mul_1:z:0(random_zoom/zoom_matrix/zeros_2:output:0,random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2 
random_zoom/zoom_matrix/concat�
random_zoom/transform/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_zoom/transform/Shape�
)random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)random_zoom/transform/strided_slice/stack�
+random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom/transform/strided_slice/stack_1�
+random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom/transform/strided_slice/stack_2�
#random_zoom/transform/strided_sliceStridedSlice$random_zoom/transform/Shape:output:02random_zoom/transform/strided_slice/stack:output:04random_zoom/transform/strided_slice/stack_1:output:04random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#random_zoom/transform/strided_slice�
0random_zoom/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2*random_flip/random_flip_left_right/add:z:0'random_zoom/zoom_matrix/concat:output:0,random_zoom/transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR22
0random_zoom/transform/ImageProjectiveTransformV2�
random_translation/ShapeShapeErandom_zoom/transform/ImageProjectiveTransformV2:transformed_images:0*
T0*
_output_shapes
:2
random_translation/Shape�
&random_translation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_translation/strided_slice/stack�
(random_translation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_translation/strided_slice/stack_1�
(random_translation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_translation/strided_slice/stack_2�
 random_translation/strided_sliceStridedSlice!random_translation/Shape:output:0/random_translation/strided_slice/stack:output:01random_translation/strided_slice/stack_1:output:01random_translation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 random_translation/strided_slice�
(random_translation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_translation/strided_slice_1/stack�
*random_translation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_translation/strided_slice_1/stack_1�
*random_translation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_translation/strided_slice_1/stack_2�
"random_translation/strided_slice_1StridedSlice!random_translation/Shape:output:01random_translation/strided_slice_1/stack:output:03random_translation/strided_slice_1/stack_1:output:03random_translation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_translation/strided_slice_1�
random_translation/CastCast+random_translation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_translation/Cast�
(random_translation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_translation/strided_slice_2/stack�
*random_translation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_translation/strided_slice_2/stack_1�
*random_translation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_translation/strided_slice_2/stack_2�
"random_translation/strided_slice_2StridedSlice!random_translation/Shape:output:01random_translation/strided_slice_2/stack:output:03random_translation/strided_slice_2/stack_1:output:03random_translation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_translation/strided_slice_2�
random_translation/Cast_1Cast+random_translation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_translation/Cast_1�
+random_translation/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+random_translation/stateful_uniform/shape/1�
)random_translation/stateful_uniform/shapePack)random_translation/strided_slice:output:04random_translation/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)random_translation/stateful_uniform/shape�
'random_translation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�2)
'random_translation/stateful_uniform/min�
'random_translation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2)
'random_translation/stateful_uniform/max�
=random_translation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2?
=random_translation/stateful_uniform/StatefulUniform/algorithm�
3random_translation/stateful_uniform/StatefulUniformStatefulUniform<random_translation_stateful_uniform_statefuluniform_resourceFrandom_translation/stateful_uniform/StatefulUniform/algorithm:output:02random_translation/stateful_uniform/shape:output:0*'
_output_shapes
:���������*
shape_dtype025
3random_translation/stateful_uniform/StatefulUniform�
'random_translation/stateful_uniform/subSub0random_translation/stateful_uniform/max:output:00random_translation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2)
'random_translation/stateful_uniform/sub�
'random_translation/stateful_uniform/mulMul<random_translation/stateful_uniform/StatefulUniform:output:0+random_translation/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:���������2)
'random_translation/stateful_uniform/mul�
#random_translation/stateful_uniformAdd+random_translation/stateful_uniform/mul:z:00random_translation/stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������2%
#random_translation/stateful_uniform�
random_translation/mulMul'random_translation/stateful_uniform:z:0random_translation/Cast:y:0*
T0*'
_output_shapes
:���������2
random_translation/mul�
-random_translation/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-random_translation/stateful_uniform_1/shape/1�
+random_translation/stateful_uniform_1/shapePack)random_translation/strided_slice:output:06random_translation/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+random_translation/stateful_uniform_1/shape�
)random_translation/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L�2+
)random_translation/stateful_uniform_1/min�
)random_translation/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2+
)random_translation/stateful_uniform_1/max�
?random_translation/stateful_uniform_1/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2A
?random_translation/stateful_uniform_1/StatefulUniform/algorithm�
5random_translation/stateful_uniform_1/StatefulUniformStatefulUniform<random_translation_stateful_uniform_statefuluniform_resourceHrandom_translation/stateful_uniform_1/StatefulUniform/algorithm:output:04random_translation/stateful_uniform_1/shape:output:04^random_translation/stateful_uniform/StatefulUniform*'
_output_shapes
:���������*
shape_dtype027
5random_translation/stateful_uniform_1/StatefulUniform�
)random_translation/stateful_uniform_1/subSub2random_translation/stateful_uniform_1/max:output:02random_translation/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2+
)random_translation/stateful_uniform_1/sub�
)random_translation/stateful_uniform_1/mulMul>random_translation/stateful_uniform_1/StatefulUniform:output:0-random_translation/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:���������2+
)random_translation/stateful_uniform_1/mul�
%random_translation/stateful_uniform_1Add-random_translation/stateful_uniform_1/mul:z:02random_translation/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:���������2'
%random_translation/stateful_uniform_1�
random_translation/mul_1Mul)random_translation/stateful_uniform_1:z:0random_translation/Cast_1:y:0*
T0*'
_output_shapes
:���������2
random_translation/mul_1�
random_translation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
random_translation/concat/axis�
random_translation/concatConcatV2random_translation/mul_1:z:0random_translation/mul:z:0'random_translation/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
random_translation/concat�
+random_translation/translation_matrix/ShapeShape"random_translation/concat:output:0*
T0*
_output_shapes
:2-
+random_translation/translation_matrix/Shape�
9random_translation/translation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_translation/translation_matrix/strided_slice/stack�
;random_translation/translation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_translation/translation_matrix/strided_slice/stack_1�
;random_translation/translation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_translation/translation_matrix/strided_slice/stack_2�
3random_translation/translation_matrix/strided_sliceStridedSlice4random_translation/translation_matrix/Shape:output:0Brandom_translation/translation_matrix/strided_slice/stack:output:0Drandom_translation/translation_matrix/strided_slice/stack_1:output:0Drandom_translation/translation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3random_translation/translation_matrix/strided_slice�
0random_translation/translation_matrix/ones/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
0random_translation/translation_matrix/ones/mul/y�
.random_translation/translation_matrix/ones/mulMul<random_translation/translation_matrix/strided_slice:output:09random_translation/translation_matrix/ones/mul/y:output:0*
T0*
_output_shapes
: 20
.random_translation/translation_matrix/ones/mul�
1random_translation/translation_matrix/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�23
1random_translation/translation_matrix/ones/Less/y�
/random_translation/translation_matrix/ones/LessLess2random_translation/translation_matrix/ones/mul:z:0:random_translation/translation_matrix/ones/Less/y:output:0*
T0*
_output_shapes
: 21
/random_translation/translation_matrix/ones/Less�
3random_translation/translation_matrix/ones/packed/1Const*
_output_shapes
: *
dtype0*
value	B :25
3random_translation/translation_matrix/ones/packed/1�
1random_translation/translation_matrix/ones/packedPack<random_translation/translation_matrix/strided_slice:output:0<random_translation/translation_matrix/ones/packed/1:output:0*
N*
T0*
_output_shapes
:23
1random_translation/translation_matrix/ones/packed�
0random_translation/translation_matrix/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?22
0random_translation/translation_matrix/ones/Const�
*random_translation/translation_matrix/onesFill:random_translation/translation_matrix/ones/packed:output:09random_translation/translation_matrix/ones/Const:output:0*
T0*'
_output_shapes
:���������2,
*random_translation/translation_matrix/ones�
1random_translation/translation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :23
1random_translation/translation_matrix/zeros/mul/y�
/random_translation/translation_matrix/zeros/mulMul<random_translation/translation_matrix/strided_slice:output:0:random_translation/translation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 21
/random_translation/translation_matrix/zeros/mul�
2random_translation/translation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�24
2random_translation/translation_matrix/zeros/Less/y�
0random_translation/translation_matrix/zeros/LessLess3random_translation/translation_matrix/zeros/mul:z:0;random_translation/translation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 22
0random_translation/translation_matrix/zeros/Less�
4random_translation/translation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_translation/translation_matrix/zeros/packed/1�
2random_translation/translation_matrix/zeros/packedPack<random_translation/translation_matrix/strided_slice:output:0=random_translation/translation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:24
2random_translation/translation_matrix/zeros/packed�
1random_translation/translation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1random_translation/translation_matrix/zeros/Const�
+random_translation/translation_matrix/zerosFill;random_translation/translation_matrix/zeros/packed:output:0:random_translation/translation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2-
+random_translation/translation_matrix/zeros�
;random_translation/translation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2=
;random_translation/translation_matrix/strided_slice_1/stack�
=random_translation/translation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2?
=random_translation/translation_matrix/strided_slice_1/stack_1�
=random_translation/translation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2?
=random_translation/translation_matrix/strided_slice_1/stack_2�
5random_translation/translation_matrix/strided_slice_1StridedSlice"random_translation/concat:output:0Drandom_translation/translation_matrix/strided_slice_1/stack:output:0Frandom_translation/translation_matrix/strided_slice_1/stack_1:output:0Frandom_translation/translation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask27
5random_translation/translation_matrix/strided_slice_1�
)random_translation/translation_matrix/NegNeg>random_translation/translation_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������2+
)random_translation/translation_matrix/Neg�
3random_translation/translation_matrix/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :25
3random_translation/translation_matrix/zeros_1/mul/y�
1random_translation/translation_matrix/zeros_1/mulMul<random_translation/translation_matrix/strided_slice:output:0<random_translation/translation_matrix/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 23
1random_translation/translation_matrix/zeros_1/mul�
4random_translation/translation_matrix/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�26
4random_translation/translation_matrix/zeros_1/Less/y�
2random_translation/translation_matrix/zeros_1/LessLess5random_translation/translation_matrix/zeros_1/mul:z:0=random_translation/translation_matrix/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 24
2random_translation/translation_matrix/zeros_1/Less�
6random_translation/translation_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :28
6random_translation/translation_matrix/zeros_1/packed/1�
4random_translation/translation_matrix/zeros_1/packedPack<random_translation/translation_matrix/strided_slice:output:0?random_translation/translation_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:26
4random_translation/translation_matrix/zeros_1/packed�
3random_translation/translation_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3random_translation/translation_matrix/zeros_1/Const�
-random_translation/translation_matrix/zeros_1Fill=random_translation/translation_matrix/zeros_1/packed:output:0<random_translation/translation_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2/
-random_translation/translation_matrix/zeros_1�
2random_translation/translation_matrix/ones_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2random_translation/translation_matrix/ones_1/mul/y�
0random_translation/translation_matrix/ones_1/mulMul<random_translation/translation_matrix/strided_slice:output:0;random_translation/translation_matrix/ones_1/mul/y:output:0*
T0*
_output_shapes
: 22
0random_translation/translation_matrix/ones_1/mul�
3random_translation/translation_matrix/ones_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�25
3random_translation/translation_matrix/ones_1/Less/y�
1random_translation/translation_matrix/ones_1/LessLess4random_translation/translation_matrix/ones_1/mul:z:0<random_translation/translation_matrix/ones_1/Less/y:output:0*
T0*
_output_shapes
: 23
1random_translation/translation_matrix/ones_1/Less�
5random_translation/translation_matrix/ones_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :27
5random_translation/translation_matrix/ones_1/packed/1�
3random_translation/translation_matrix/ones_1/packedPack<random_translation/translation_matrix/strided_slice:output:0>random_translation/translation_matrix/ones_1/packed/1:output:0*
N*
T0*
_output_shapes
:25
3random_translation/translation_matrix/ones_1/packed�
2random_translation/translation_matrix/ones_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?24
2random_translation/translation_matrix/ones_1/Const�
,random_translation/translation_matrix/ones_1Fill<random_translation/translation_matrix/ones_1/packed:output:0;random_translation/translation_matrix/ones_1/Const:output:0*
T0*'
_output_shapes
:���������2.
,random_translation/translation_matrix/ones_1�
;random_translation/translation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2=
;random_translation/translation_matrix/strided_slice_2/stack�
=random_translation/translation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2?
=random_translation/translation_matrix/strided_slice_2/stack_1�
=random_translation/translation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2?
=random_translation/translation_matrix/strided_slice_2/stack_2�
5random_translation/translation_matrix/strided_slice_2StridedSlice"random_translation/concat:output:0Drandom_translation/translation_matrix/strided_slice_2/stack:output:0Frandom_translation/translation_matrix/strided_slice_2/stack_1:output:0Frandom_translation/translation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask27
5random_translation/translation_matrix/strided_slice_2�
+random_translation/translation_matrix/Neg_1Neg>random_translation/translation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2-
+random_translation/translation_matrix/Neg_1�
3random_translation/translation_matrix/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :25
3random_translation/translation_matrix/zeros_2/mul/y�
1random_translation/translation_matrix/zeros_2/mulMul<random_translation/translation_matrix/strided_slice:output:0<random_translation/translation_matrix/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 23
1random_translation/translation_matrix/zeros_2/mul�
4random_translation/translation_matrix/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�26
4random_translation/translation_matrix/zeros_2/Less/y�
2random_translation/translation_matrix/zeros_2/LessLess5random_translation/translation_matrix/zeros_2/mul:z:0=random_translation/translation_matrix/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 24
2random_translation/translation_matrix/zeros_2/Less�
6random_translation/translation_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :28
6random_translation/translation_matrix/zeros_2/packed/1�
4random_translation/translation_matrix/zeros_2/packedPack<random_translation/translation_matrix/strided_slice:output:0?random_translation/translation_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:26
4random_translation/translation_matrix/zeros_2/packed�
3random_translation/translation_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3random_translation/translation_matrix/zeros_2/Const�
-random_translation/translation_matrix/zeros_2Fill=random_translation/translation_matrix/zeros_2/packed:output:0<random_translation/translation_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������2/
-random_translation/translation_matrix/zeros_2�
1random_translation/translation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :23
1random_translation/translation_matrix/concat/axis�
,random_translation/translation_matrix/concatConcatV23random_translation/translation_matrix/ones:output:04random_translation/translation_matrix/zeros:output:0-random_translation/translation_matrix/Neg:y:06random_translation/translation_matrix/zeros_1:output:05random_translation/translation_matrix/ones_1:output:0/random_translation/translation_matrix/Neg_1:y:06random_translation/translation_matrix/zeros_2:output:0:random_translation/translation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2.
,random_translation/translation_matrix/concat�
"random_translation/transform/ShapeShapeErandom_zoom/transform/ImageProjectiveTransformV2:transformed_images:0*
T0*
_output_shapes
:2$
"random_translation/transform/Shape�
0random_translation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0random_translation/transform/strided_slice/stack�
2random_translation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2random_translation/transform/strided_slice/stack_1�
2random_translation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2random_translation/transform/strided_slice/stack_2�
*random_translation/transform/strided_sliceStridedSlice+random_translation/transform/Shape:output:09random_translation/transform/strided_slice/stack:output:0;random_translation/transform/strided_slice/stack_1:output:0;random_translation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*random_translation/transform/strided_slice�
7random_translation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2Erandom_zoom/transform/ImageProjectiveTransformV2:transformed_images:05random_translation/translation_matrix/concat:output:03random_translation/transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
interpolation
BILINEAR29
7random_translation/transform/ImageProjectiveTransformV2�
#encoder_conv1/Conv2D/ReadVariableOpReadVariableOp,encoder_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02%
#encoder_conv1/Conv2D/ReadVariableOp�
encoder_conv1/Conv2DConv2DLrandom_translation/transform/ImageProjectiveTransformV2:transformed_images:0+encoder_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
encoder_conv1/Conv2D�
$encoder_conv1/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder_conv1/BiasAdd/ReadVariableOp�
encoder_conv1/BiasAddBiasAddencoder_conv1/Conv2D:output:0,encoder_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
encoder_conv1/BiasAdd�
encoder_conv1/EluEluencoder_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
encoder_conv1/Elu�
encoder_pool1/MaxPoolMaxPoolencoder_conv1/Elu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingSAME*
strides
2
encoder_pool1/MaxPool�
encoder_dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
encoder_dropout1/dropout/Const�
encoder_dropout1/dropout/MulMulencoder_pool1/MaxPool:output:0'encoder_dropout1/dropout/Const:output:0*
T0*/
_output_shapes
:���������@@2
encoder_dropout1/dropout/Mul�
encoder_dropout1/dropout/ShapeShapeencoder_pool1/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout1/dropout/Shape�
5encoder_dropout1/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout1/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype027
5encoder_dropout1/dropout/random_uniform/RandomUniform�
'encoder_dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2)
'encoder_dropout1/dropout/GreaterEqual/y�
%encoder_dropout1/dropout/GreaterEqualGreaterEqual>encoder_dropout1/dropout/random_uniform/RandomUniform:output:00encoder_dropout1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@@2'
%encoder_dropout1/dropout/GreaterEqual�
encoder_dropout1/dropout/CastCast)encoder_dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2
encoder_dropout1/dropout/Cast�
encoder_dropout1/dropout/Mul_1Mul encoder_dropout1/dropout/Mul:z:0!encoder_dropout1/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2 
encoder_dropout1/dropout/Mul_1�
#encoder_conv2/Conv2D/ReadVariableOpReadVariableOp,encoder_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#encoder_conv2/Conv2D/ReadVariableOp�
encoder_conv2/Conv2DConv2D"encoder_dropout1/dropout/Mul_1:z:0+encoder_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
2
encoder_conv2/Conv2D�
$encoder_conv2/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$encoder_conv2/BiasAdd/ReadVariableOp�
encoder_conv2/BiasAddBiasAddencoder_conv2/Conv2D:output:0,encoder_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
encoder_conv2/BiasAdd�
encoder_conv2/EluEluencoder_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ 2
encoder_conv2/Elu�
encoder_pool2/MaxPoolMaxPoolencoder_conv2/Elu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingSAME*
strides
2
encoder_pool2/MaxPool�
encoder_dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
encoder_dropout2/dropout/Const�
encoder_dropout2/dropout/MulMulencoder_pool2/MaxPool:output:0'encoder_dropout2/dropout/Const:output:0*
T0*/
_output_shapes
:���������   2
encoder_dropout2/dropout/Mul�
encoder_dropout2/dropout/ShapeShapeencoder_pool2/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout2/dropout/Shape�
5encoder_dropout2/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout2/dropout/Shape:output:0*
T0*/
_output_shapes
:���������   *
dtype027
5encoder_dropout2/dropout/random_uniform/RandomUniform�
'encoder_dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2)
'encoder_dropout2/dropout/GreaterEqual/y�
%encoder_dropout2/dropout/GreaterEqualGreaterEqual>encoder_dropout2/dropout/random_uniform/RandomUniform:output:00encoder_dropout2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������   2'
%encoder_dropout2/dropout/GreaterEqual�
encoder_dropout2/dropout/CastCast)encoder_dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������   2
encoder_dropout2/dropout/Cast�
encoder_dropout2/dropout/Mul_1Mul encoder_dropout2/dropout/Mul:z:0!encoder_dropout2/dropout/Cast:y:0*
T0*/
_output_shapes
:���������   2 
encoder_dropout2/dropout/Mul_1�
#encoder_conv3/Conv2D/ReadVariableOpReadVariableOp,encoder_conv3_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02%
#encoder_conv3/Conv2D/ReadVariableOp�
encoder_conv3/Conv2DConv2D"encoder_dropout2/dropout/Mul_1:z:0+encoder_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  0*
paddingSAME*
strides
2
encoder_conv3/Conv2D�
$encoder_conv3/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$encoder_conv3/BiasAdd/ReadVariableOp�
encoder_conv3/BiasAddBiasAddencoder_conv3/Conv2D:output:0,encoder_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  02
encoder_conv3/BiasAdd�
encoder_conv3/EluEluencoder_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:���������  02
encoder_conv3/Elu�
encoder_pool3/MaxPoolMaxPoolencoder_conv3/Elu:activations:0*/
_output_shapes
:���������0*
ksize
*
paddingSAME*
strides
2
encoder_pool3/MaxPool�
encoder_dropout3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
encoder_dropout3/dropout/Const�
encoder_dropout3/dropout/MulMulencoder_pool3/MaxPool:output:0'encoder_dropout3/dropout/Const:output:0*
T0*/
_output_shapes
:���������02
encoder_dropout3/dropout/Mul�
encoder_dropout3/dropout/ShapeShapeencoder_pool3/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout3/dropout/Shape�
5encoder_dropout3/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout3/dropout/Shape:output:0*
T0*/
_output_shapes
:���������0*
dtype027
5encoder_dropout3/dropout/random_uniform/RandomUniform�
'encoder_dropout3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2)
'encoder_dropout3/dropout/GreaterEqual/y�
%encoder_dropout3/dropout/GreaterEqualGreaterEqual>encoder_dropout3/dropout/random_uniform/RandomUniform:output:00encoder_dropout3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������02'
%encoder_dropout3/dropout/GreaterEqual�
encoder_dropout3/dropout/CastCast)encoder_dropout3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������02
encoder_dropout3/dropout/Cast�
encoder_dropout3/dropout/Mul_1Mul encoder_dropout3/dropout/Mul:z:0!encoder_dropout3/dropout/Cast:y:0*
T0*/
_output_shapes
:���������02 
encoder_dropout3/dropout/Mul_1�
#encoder_conv4/Conv2D/ReadVariableOpReadVariableOp,encoder_conv4_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype02%
#encoder_conv4/Conv2D/ReadVariableOp�
encoder_conv4/Conv2DConv2D"encoder_dropout3/dropout/Mul_1:z:0+encoder_conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
encoder_conv4/Conv2D�
$encoder_conv4/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$encoder_conv4/BiasAdd/ReadVariableOp�
encoder_conv4/BiasAddBiasAddencoder_conv4/Conv2D:output:0,encoder_conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
encoder_conv4/BiasAdd�
encoder_conv4/EluEluencoder_conv4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
encoder_conv4/Elu�
encoder_pool4/MaxPoolMaxPoolencoder_conv4/Elu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingSAME*
strides
2
encoder_pool4/MaxPool�
encoder_dropout4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
encoder_dropout4/dropout/Const�
encoder_dropout4/dropout/MulMulencoder_pool4/MaxPool:output:0'encoder_dropout4/dropout/Const:output:0*
T0*/
_output_shapes
:���������@2
encoder_dropout4/dropout/Mul�
encoder_dropout4/dropout/ShapeShapeencoder_pool4/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout4/dropout/Shape�
5encoder_dropout4/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout4/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype027
5encoder_dropout4/dropout/random_uniform/RandomUniform�
'encoder_dropout4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2)
'encoder_dropout4/dropout/GreaterEqual/y�
%encoder_dropout4/dropout/GreaterEqualGreaterEqual>encoder_dropout4/dropout/random_uniform/RandomUniform:output:00encoder_dropout4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2'
%encoder_dropout4/dropout/GreaterEqual�
encoder_dropout4/dropout/CastCast)encoder_dropout4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
encoder_dropout4/dropout/Cast�
encoder_dropout4/dropout/Mul_1Mul encoder_dropout4/dropout/Mul:z:0!encoder_dropout4/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2 
encoder_dropout4/dropout/Mul_1�
#encoder_conv5/Conv2D/ReadVariableOpReadVariableOp,encoder_conv5_conv2d_readvariableop_resource*&
_output_shapes
:@P*
dtype02%
#encoder_conv5/Conv2D/ReadVariableOp�
encoder_conv5/Conv2DConv2D"encoder_dropout4/dropout/Mul_1:z:0+encoder_conv5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P*
paddingSAME*
strides
2
encoder_conv5/Conv2D�
$encoder_conv5/BiasAdd/ReadVariableOpReadVariableOp-encoder_conv5_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder_conv5/BiasAdd/ReadVariableOp�
encoder_conv5/BiasAddBiasAddencoder_conv5/Conv2D:output:0,encoder_conv5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P2
encoder_conv5/BiasAdd�
encoder_conv5/EluEluencoder_conv5/BiasAdd:output:0*
T0*/
_output_shapes
:���������P2
encoder_conv5/Elu�
encoder_pool5/MaxPoolMaxPoolencoder_conv5/Elu:activations:0*/
_output_shapes
:���������P*
ksize
*
paddingSAME*
strides
2
encoder_pool5/MaxPool�
encoder_dropout5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
encoder_dropout5/dropout/Const�
encoder_dropout5/dropout/MulMulencoder_pool5/MaxPool:output:0'encoder_dropout5/dropout/Const:output:0*
T0*/
_output_shapes
:���������P2
encoder_dropout5/dropout/Mul�
encoder_dropout5/dropout/ShapeShapeencoder_pool5/MaxPool:output:0*
T0*
_output_shapes
:2 
encoder_dropout5/dropout/Shape�
5encoder_dropout5/dropout/random_uniform/RandomUniformRandomUniform'encoder_dropout5/dropout/Shape:output:0*
T0*/
_output_shapes
:���������P*
dtype027
5encoder_dropout5/dropout/random_uniform/RandomUniform�
'encoder_dropout5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2)
'encoder_dropout5/dropout/GreaterEqual/y�
%encoder_dropout5/dropout/GreaterEqualGreaterEqual>encoder_dropout5/dropout/random_uniform/RandomUniform:output:00encoder_dropout5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������P2'
%encoder_dropout5/dropout/GreaterEqual�
encoder_dropout5/dropout/CastCast)encoder_dropout5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������P2
encoder_dropout5/dropout/Cast�
encoder_dropout5/dropout/Mul_1Mul encoder_dropout5/dropout/Mul:z:0!encoder_dropout5/dropout/Cast:y:0*
T0*/
_output_shapes
:���������P2 
encoder_dropout5/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshape"encoder_dropout5/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:����������
2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense/Sigmoid�
IdentityIdentitydense/Sigmoid:y:04^random_translation/stateful_uniform/StatefulUniform6^random_translation/stateful_uniform_1/StatefulUniform-^random_zoom/stateful_uniform/StatefulUniform/^random_zoom/stateful_uniform_1/StatefulUniform*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:�����������::::::::::::::2j
3random_translation/stateful_uniform/StatefulUniform3random_translation/stateful_uniform/StatefulUniform2n
5random_translation/stateful_uniform_1/StatefulUniform5random_translation/stateful_uniform_1/StatefulUniform2\
,random_zoom/stateful_uniform/StatefulUniform,random_zoom/stateful_uniform/StatefulUniform2`
.random_zoom/stateful_uniform_1/StatefulUniform.random_zoom/stateful_uniform_1/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_14265

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Eluo
IdentityIdentityElu:activations:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������:::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_13911

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
j
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_14410

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������02
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������02
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
L
0__inference_encoder_dropout3_layer_call_fn_17029

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_144152
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_14550

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P:W S
/
_output_shapes
:���������P
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_16465

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

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_146892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:�����������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������
E
input_2:
serving_default_input_2:0�����������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"��
_tf_keras_network��{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomTranslation", "config": {"name": "random_translation", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Abs", "trainable": true, "dtype": "float32", "node_def": {"name": "Abs", "op": "Abs", "input": ["subtract/sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Abs", "inbound_nodes": [[["subtract", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["tf_op_layer_Abs", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 1]}, {"class_name": "TensorShape", "items": [null, 128, 128, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomTranslation", "config": {"name": "random_translation", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "Subtract", "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "name": "subtract", "inbound_nodes": [[["sequential", 1, 0, {}], ["sequential", 2, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Abs", "trainable": true, "dtype": "float32", "node_def": {"name": "Abs", "op": "Abs", "input": ["subtract/sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Abs", "inbound_nodes": [[["subtract", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["tf_op_layer_Abs", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["AUC"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0003499999875202775, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�w
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
layer-7
layer-8
layer_with_weights-2
layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
 layer_with_weights-5
 layer-19
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�s
_tf_keras_sequential�s{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomTranslation", "config": {"name": "random_translation", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomZoom", "config": {"name": "random_zoom", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomTranslation", "config": {"name": "random_translation", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Subtract", "name": "subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "subtract", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128]}, {"class_name": "TensorShape", "items": [null, 128]}]}
�
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Abs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Abs", "trainable": true, "dtype": "float32", "node_def": {"name": "Abs", "op": "Abs", "input": ["subtract/sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
	3decay
4learning_rate
5momentum
6rho
7iter
-rms�
.rms�
8rms�
9rms�
:rms�
;rms�
<rms�
=rms�
>rms�
?rms�
@rms�
Arms�
Brms�
Crms�"
	optimizer
�
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
-12
.13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
-12
.13"
trackable_list_wrapper
�
Dmetrics
Elayer_metrics
Fnon_trainable_variables

Glayers
Hlayer_regularization_losses
trainable_variables
	regularization_losses

	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
I_rng
J_inbound_nodes
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "RandomFlip", "name": "random_flip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_flip", "trainable": true, "dtype": "float32", "mode": "horizontal", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
O_rng
P_inbound_nodes
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "RandomZoom", "name": "random_zoom", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_zoom", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}
�
U_rng
V_inbound_nodes
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "RandomTranslation", "name": "random_translation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_translation", "trainable": true, "dtype": "float32", "height_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "width_factor": {"class_name": "__tuple__", "items": [-0.2, 0.2]}, "fill_mode": "constant", "interpolation": "bilinear", "seed": null}}
�

[_inbound_nodes

8kernel
9bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encoder_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
�
`_inbound_nodes
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "encoder_pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
e_inbound_nodes
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "encoder_dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

j_inbound_nodes

:kernel
;bias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encoder_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
�
o_inbound_nodes
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "encoder_pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
t_inbound_nodes
utrainable_variables
vregularization_losses
w	variables
x	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "encoder_dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

y_inbound_nodes

<kernel
=bias
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encoder_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
~_inbound_nodes
trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "encoder_pool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "encoder_dropout3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

�_inbound_nodes

>kernel
?bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encoder_conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 48]}}
�
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "encoder_pool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "encoder_dropout4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

�_inbound_nodes

@kernel
Abias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "encoder_conv5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_conv5", "trainable": true, "dtype": "float32", "filters": 80, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
�
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "encoder_pool5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "encoder_dropout5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder_dropout5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�
�_inbound_nodes
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�_inbound_nodes

Bkernel
Cbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1280}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1280]}}
v
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
!trainable_variables
"regularization_losses
#	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
%trainable_variables
&regularization_losses
'	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
)trainable_variables
*regularization_losses
+	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_1/kernel
:2dense_1/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
/trainable_variables
0regularization_losses
1	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
: (2rho
:	 (2RMSprop/iter
9:72sequential/encoder_conv1/kernel
+:)2sequential/encoder_conv1/bias
9:7 2sequential/encoder_conv2/kernel
+:) 2sequential/encoder_conv2/bias
9:7 02sequential/encoder_conv3/kernel
+:)02sequential/encoder_conv3/bias
9:70@2sequential/encoder_conv4/kernel
+:)@2sequential/encoder_conv4/bias
9:7@P2sequential/encoder_conv5/kernel
+:)P2sequential/encoder_conv5/bias
+:)
�
�2sequential/dense/kernel
$:"�2sequential/dense/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
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
/
�
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
Ktrainable_variables
Lregularization_losses
M	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/
�
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
Qtrainable_variables
Rregularization_losses
S	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/
�
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
Wtrainable_variables
Xregularization_losses
Y	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
\trainable_variables
]regularization_losses
^	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
atrainable_variables
bregularization_losses
c	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ftrainable_variables
gregularization_losses
h	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ktrainable_variables
lregularization_losses
m	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ptrainable_variables
qregularization_losses
r	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
utrainable_variables
vregularization_losses
w	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
ztrainable_variables
{regularization_losses
|	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
�
�metrics
�layer_metrics
�non_trainable_variables
�layers
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
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
17
18
 19"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�"
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api"�!
_tf_keras_metric�!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:	2Variable
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
:	2Variable
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
:	2Variable
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
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
+:)	�2RMSprop/dense_1/kernel/rms
$:"2RMSprop/dense_1/bias/rms
C:A2+RMSprop/sequential/encoder_conv1/kernel/rms
5:32)RMSprop/sequential/encoder_conv1/bias/rms
C:A 2+RMSprop/sequential/encoder_conv2/kernel/rms
5:3 2)RMSprop/sequential/encoder_conv2/bias/rms
C:A 02+RMSprop/sequential/encoder_conv3/kernel/rms
5:302)RMSprop/sequential/encoder_conv3/bias/rms
C:A0@2+RMSprop/sequential/encoder_conv4/kernel/rms
5:3@2)RMSprop/sequential/encoder_conv4/bias/rms
C:A@P2+RMSprop/sequential/encoder_conv5/kernel/rms
5:3P2)RMSprop/sequential/encoder_conv5/bias/rms
5:3
�
�2#RMSprop/sequential/dense/kernel/rms
.:,�2!RMSprop/sequential/dense/bias/rms
�2�
G__inference_functional_1_layer_call_and_return_conditional_losses_15014
G__inference_functional_1_layer_call_and_return_conditional_losses_14964
G__inference_functional_1_layer_call_and_return_conditional_losses_15981
G__inference_functional_1_layer_call_and_return_conditional_losses_15871�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_13769�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *b�_
]�Z
+�(
input_1�����������
+�(
input_2�����������
�2�
,__inference_functional_1_layer_call_fn_16019
,__inference_functional_1_layer_call_fn_16053
,__inference_functional_1_layer_call_fn_15193
,__inference_functional_1_layer_call_fn_15109�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_sequential_layer_call_and_return_conditional_losses_16374
E__inference_sequential_layer_call_and_return_conditional_losses_14634
E__inference_sequential_layer_call_and_return_conditional_losses_14586
E__inference_sequential_layer_call_and_return_conditional_losses_16432�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_sequential_layer_call_fn_14720
*__inference_sequential_layer_call_fn_14797
*__inference_sequential_layer_call_fn_16465
*__inference_sequential_layer_call_fn_16494�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_subtract_layer_call_and_return_conditional_losses_16500�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_subtract_layer_call_fn_16506�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_16511�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_tf_op_layer_Abs_layer_call_fn_16516�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_16527�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_16536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
9B7
#__inference_signature_wrapper_15237input_1input_2
�2�
F__inference_random_flip_layer_call_and_return_conditional_losses_16588
F__inference_random_flip_layer_call_and_return_conditional_losses_16630
F__inference_random_flip_layer_call_and_return_conditional_losses_16592
F__inference_random_flip_layer_call_and_return_conditional_losses_16634�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_random_flip_layer_call_fn_16597
+__inference_random_flip_layer_call_fn_16644
+__inference_random_flip_layer_call_fn_16639
+__inference_random_flip_layer_call_fn_16602�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_random_zoom_layer_call_and_return_conditional_losses_16751
F__inference_random_zoom_layer_call_and_return_conditional_losses_16755�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_random_zoom_layer_call_fn_16762
+__inference_random_zoom_layer_call_fn_16767�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_random_translation_layer_call_and_return_conditional_losses_16876
M__inference_random_translation_layer_call_and_return_conditional_losses_16872�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_random_translation_layer_call_fn_16883
2__inference_random_translation_layer_call_fn_16888�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_16899�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_encoder_conv1_layer_call_fn_16908�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_13899�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_encoder_pool1_layer_call_fn_13905�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_16920
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_16925�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_encoder_dropout1_layer_call_fn_16930
0__inference_encoder_dropout1_layer_call_fn_16935�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_16946�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_encoder_conv2_layer_call_fn_16955�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_13911�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_encoder_pool2_layer_call_fn_13917�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_16967
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_16972�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_encoder_dropout2_layer_call_fn_16977
0__inference_encoder_dropout2_layer_call_fn_16982�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_16993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_encoder_conv3_layer_call_fn_17002�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_13923�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_encoder_pool3_layer_call_fn_13929�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17019
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17014�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_encoder_dropout3_layer_call_fn_17024
0__inference_encoder_dropout3_layer_call_fn_17029�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_17040�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_encoder_conv4_layer_call_fn_17049�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_13935�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_encoder_pool4_layer_call_fn_13941�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17066
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17061�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_encoder_dropout4_layer_call_fn_17071
0__inference_encoder_dropout4_layer_call_fn_17076�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_17087�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_encoder_conv5_layer_call_fn_17096�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_13947�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_encoder_pool5_layer_call_fn_13953�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17113
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17108�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_encoder_dropout5_layer_call_fn_17118
0__inference_encoder_dropout5_layer_call_fn_17123�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_17129�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_17134�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_17145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_17154�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_13769�89:;<=>?@ABC-.l�i
b�_
]�Z
+�(
input_1�����������
+�(
input_2�����������
� "1�.
,
dense_1!�
dense_1����������
B__inference_dense_1_layer_call_and_return_conditional_losses_16527]-.0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_dense_1_layer_call_fn_16536P-.0�-
&�#
!�
inputs����������
� "�����������
@__inference_dense_layer_call_and_return_conditional_losses_17145^BC0�-
&�#
!�
inputs����������

� "&�#
�
0����������
� z
%__inference_dense_layer_call_fn_17154QBC0�-
&�#
!�
inputs����������

� "������������
H__inference_encoder_conv1_layer_call_and_return_conditional_losses_16899p899�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
-__inference_encoder_conv1_layer_call_fn_16908c899�6
/�,
*�'
inputs�����������
� ""�������������
H__inference_encoder_conv2_layer_call_and_return_conditional_losses_16946l:;7�4
-�*
(�%
inputs���������@@
� "-�*
#� 
0���������@@ 
� �
-__inference_encoder_conv2_layer_call_fn_16955_:;7�4
-�*
(�%
inputs���������@@
� " ����������@@ �
H__inference_encoder_conv3_layer_call_and_return_conditional_losses_16993l<=7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������  0
� �
-__inference_encoder_conv3_layer_call_fn_17002_<=7�4
-�*
(�%
inputs���������   
� " ����������  0�
H__inference_encoder_conv4_layer_call_and_return_conditional_losses_17040l>?7�4
-�*
(�%
inputs���������0
� "-�*
#� 
0���������@
� �
-__inference_encoder_conv4_layer_call_fn_17049_>?7�4
-�*
(�%
inputs���������0
� " ����������@�
H__inference_encoder_conv5_layer_call_and_return_conditional_losses_17087l@A7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������P
� �
-__inference_encoder_conv5_layer_call_fn_17096_@A7�4
-�*
(�%
inputs���������@
� " ����������P�
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_16920l;�8
1�.
(�%
inputs���������@@
p
� "-�*
#� 
0���������@@
� �
K__inference_encoder_dropout1_layer_call_and_return_conditional_losses_16925l;�8
1�.
(�%
inputs���������@@
p 
� "-�*
#� 
0���������@@
� �
0__inference_encoder_dropout1_layer_call_fn_16930_;�8
1�.
(�%
inputs���������@@
p
� " ����������@@�
0__inference_encoder_dropout1_layer_call_fn_16935_;�8
1�.
(�%
inputs���������@@
p 
� " ����������@@�
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_16967l;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
K__inference_encoder_dropout2_layer_call_and_return_conditional_losses_16972l;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
0__inference_encoder_dropout2_layer_call_fn_16977_;�8
1�.
(�%
inputs���������   
p
� " ����������   �
0__inference_encoder_dropout2_layer_call_fn_16982_;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17014l;�8
1�.
(�%
inputs���������0
p
� "-�*
#� 
0���������0
� �
K__inference_encoder_dropout3_layer_call_and_return_conditional_losses_17019l;�8
1�.
(�%
inputs���������0
p 
� "-�*
#� 
0���������0
� �
0__inference_encoder_dropout3_layer_call_fn_17024_;�8
1�.
(�%
inputs���������0
p
� " ����������0�
0__inference_encoder_dropout3_layer_call_fn_17029_;�8
1�.
(�%
inputs���������0
p 
� " ����������0�
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17061l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
K__inference_encoder_dropout4_layer_call_and_return_conditional_losses_17066l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_encoder_dropout4_layer_call_fn_17071_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
0__inference_encoder_dropout4_layer_call_fn_17076_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17108l;�8
1�.
(�%
inputs���������P
p
� "-�*
#� 
0���������P
� �
K__inference_encoder_dropout5_layer_call_and_return_conditional_losses_17113l;�8
1�.
(�%
inputs���������P
p 
� "-�*
#� 
0���������P
� �
0__inference_encoder_dropout5_layer_call_fn_17118_;�8
1�.
(�%
inputs���������P
p
� " ����������P�
0__inference_encoder_dropout5_layer_call_fn_17123_;�8
1�.
(�%
inputs���������P
p 
� " ����������P�
H__inference_encoder_pool1_layer_call_and_return_conditional_losses_13899�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_encoder_pool1_layer_call_fn_13905�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
H__inference_encoder_pool2_layer_call_and_return_conditional_losses_13911�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_encoder_pool2_layer_call_fn_13917�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
H__inference_encoder_pool3_layer_call_and_return_conditional_losses_13923�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_encoder_pool3_layer_call_fn_13929�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
H__inference_encoder_pool4_layer_call_and_return_conditional_losses_13935�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_encoder_pool4_layer_call_fn_13941�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
H__inference_encoder_pool5_layer_call_and_return_conditional_losses_13947�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_encoder_pool5_layer_call_fn_13953�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
B__inference_flatten_layer_call_and_return_conditional_losses_17129a7�4
-�*
(�%
inputs���������P
� "&�#
�
0����������

� 
'__inference_flatten_layer_call_fn_17134T7�4
-�*
(�%
inputs���������P
� "�����������
�
G__inference_functional_1_layer_call_and_return_conditional_losses_14964���89:;<=>?@ABC-.t�q
j�g
]�Z
+�(
input_1�����������
+�(
input_2�����������
p

 
� "%�"
�
0���������
� �
G__inference_functional_1_layer_call_and_return_conditional_losses_15014�89:;<=>?@ABC-.t�q
j�g
]�Z
+�(
input_1�����������
+�(
input_2�����������
p 

 
� "%�"
�
0���������
� �
G__inference_functional_1_layer_call_and_return_conditional_losses_15871���89:;<=>?@ABC-.v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p

 
� "%�"
�
0���������
� �
G__inference_functional_1_layer_call_and_return_conditional_losses_15981�89:;<=>?@ABC-.v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p 

 
� "%�"
�
0���������
� �
,__inference_functional_1_layer_call_fn_15109���89:;<=>?@ABC-.t�q
j�g
]�Z
+�(
input_1�����������
+�(
input_2�����������
p

 
� "�����������
,__inference_functional_1_layer_call_fn_15193�89:;<=>?@ABC-.t�q
j�g
]�Z
+�(
input_1�����������
+�(
input_2�����������
p 

 
� "�����������
,__inference_functional_1_layer_call_fn_16019���89:;<=>?@ABC-.v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p

 
� "�����������
,__inference_functional_1_layer_call_fn_16053�89:;<=>?@ABC-.v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p 

 
� "�����������
F__inference_random_flip_layer_call_and_return_conditional_losses_16588�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
F__inference_random_flip_layer_call_and_return_conditional_losses_16592�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
F__inference_random_flip_layer_call_and_return_conditional_losses_16630p=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
F__inference_random_flip_layer_call_and_return_conditional_losses_16634p=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
+__inference_random_flip_layer_call_fn_16597�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
+__inference_random_flip_layer_call_fn_16602�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
+__inference_random_flip_layer_call_fn_16639c=�:
3�0
*�'
inputs�����������
p
� ""�������������
+__inference_random_flip_layer_call_fn_16644c=�:
3�0
*�'
inputs�����������
p 
� ""�������������
M__inference_random_translation_layer_call_and_return_conditional_losses_16872t�=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
M__inference_random_translation_layer_call_and_return_conditional_losses_16876p=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
2__inference_random_translation_layer_call_fn_16883g�=�:
3�0
*�'
inputs�����������
p
� ""�������������
2__inference_random_translation_layer_call_fn_16888c=�:
3�0
*�'
inputs�����������
p 
� ""�������������
F__inference_random_zoom_layer_call_and_return_conditional_losses_16751t�=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
F__inference_random_zoom_layer_call_and_return_conditional_losses_16755p=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
+__inference_random_zoom_layer_call_fn_16762g�=�:
3�0
*�'
inputs�����������
p
� ""�������������
+__inference_random_zoom_layer_call_fn_16767c=�:
3�0
*�'
inputs�����������
p 
� ""�������������
E__inference_sequential_layer_call_and_return_conditional_losses_14586���89:;<=>?@ABCL�I
B�?
5�2
random_flip_input�����������
p

 
� "&�#
�
0����������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_14634�89:;<=>?@ABCL�I
B�?
5�2
random_flip_input�����������
p 

 
� "&�#
�
0����������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_16374}��89:;<=>?@ABCA�>
7�4
*�'
inputs�����������
p

 
� "&�#
�
0����������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_16432y89:;<=>?@ABCA�>
7�4
*�'
inputs�����������
p 

 
� "&�#
�
0����������
� �
*__inference_sequential_layer_call_fn_14720{��89:;<=>?@ABCL�I
B�?
5�2
random_flip_input�����������
p

 
� "������������
*__inference_sequential_layer_call_fn_14797w89:;<=>?@ABCL�I
B�?
5�2
random_flip_input�����������
p 

 
� "������������
*__inference_sequential_layer_call_fn_16465p��89:;<=>?@ABCA�>
7�4
*�'
inputs�����������
p

 
� "������������
*__inference_sequential_layer_call_fn_16494l89:;<=>?@ABCA�>
7�4
*�'
inputs�����������
p 

 
� "������������
#__inference_signature_wrapper_15237�89:;<=>?@ABC-.}�z
� 
s�p
6
input_1+�(
input_1�����������
6
input_2+�(
input_2�����������"1�.
,
dense_1!�
dense_1����������
C__inference_subtract_layer_call_and_return_conditional_losses_16500�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
(__inference_subtract_layer_call_fn_16506y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
J__inference_tf_op_layer_Abs_layer_call_and_return_conditional_losses_16511Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
/__inference_tf_op_layer_Abs_layer_call_fn_16516M0�-
&�#
!�
inputs����������
� "�����������