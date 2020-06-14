from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

y_target = 63

# 加载日志数据
ea1 = event_accumulator.EventAccumulator(
    r'./logs/mpc_control_double_model/20200613042451__y_target=[63, 32]__noise__T=0.1666667__W=1__C1==2__C2==2__size=40_noise_pre=1/events.out.tfevents.1592022291.5fa9205ac581')
ea1.Reload()
ea2 = event_accumulator.EventAccumulator(
    r'./logs/mpc_control_double_model/20200613052808__y_target=[63, 32]__noise__T=0.1666667__W=1__C1==2__C2==2__size=40_noise_pre=0/events.out.tfevents.1592026088.5fa9205ac581')
ea2.Reload()
print(ea1.scalars.Keys())

for item in ea1.scalars.Keys():
    list_1 = [i.value for i in ea1.scalars.Items(item)]
    list_2 = [i.value for i in ea2.scalars.Items(item)]
    plt.title(item)
    plt.plot(list_1, label='noise_pre=1')
    plt.plot(list_2, label='noise_pre=0')
    if item == 'State/Concentration_of_underﬂow':
        plt.plot([y_target]*len(list_1), label='set_point='+str(y_target), color='black')
    plt.xlabel('min')
    plt.legend()
    plt.show()

# concentration_list_1 = [i.value for i in ea1.scalars.Items('State/Concentration_of_underﬂow')]
# concentration_list_2 = [i.value for i in ea2.scalars.Items('State/Concentration_of_underﬂow')]
#
# cost_list_1 = [i.value for i in ea1.scalars.Items('Cost')]
# cost_list_2 = [i.value for i in ea2.scalars.Items('Cost')]
#
#
# plt.plot(concentration_list_1, label='noise_pre=1')
# plt.plot(concentration_list_2, label='noise_pre=0')
# plt.xlabel('min')
# plt.ylabel('Concentration')
# plt.legend()
# plt.show()

