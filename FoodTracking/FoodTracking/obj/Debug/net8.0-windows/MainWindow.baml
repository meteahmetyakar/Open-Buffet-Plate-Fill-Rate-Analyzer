   M S B A M L   `   `   `  ����U  QLiveCharts.Wpf, Version=0.9.7.0, Culture=neutral, PublicKeyToken=0bc1f845d1ebb8dfG4clr-namespace:LiveCharts.Wpf;assembly=LiveCharts.WpfLiveCharts.Wpf  5   	   G CFoodTracking, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null+clr-namespace:FoodTrackingFoodTracking 5   	      FoodTracking.MainWindow   5   	   W SPresentationCore, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5   	   R NSystem.Xaml, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   Dlvc4clr-namespace:LiveCharts.Wpf;assembly=LiveCharts.Wpf     5   	   ,localclr-namespace:FoodTracking     5   	     � Title$  
MainWindow��5	   	   $	��900��6   $
��1200��6)   .��5
      �� 
�������$��2q�6    $��10q�64   ._�5   
   � -   5         �Name$ MainGrid��5      ���	#FFD9D9D9.+�5      �� 
�������$��1q�6(   $��36,28,36,313q�6<   ���	#FFF8EAEA �� CornerRadius$ 	20,20,0,0��5      ._�5      � -   5        $ PieCharGrid��5      $��9,9,9,9q�6,   ���	#FFF8EAEA � RowDefinitions 5      �� 5      $��* �6(   �� 5      $��* �6(    � ColumnDefinitions 5      �� 5      $��* �6+   �� 5      $��* �6+   �� 5      $��* �6+   �� 5       $��* �6+   .+�5$      �� $��0��6   $��0��6*   
�������$��1q�6M    �� Padding$ 10q�6a   ._�5%      � .+�5&         LiveCharts.Wpf.PieChart  -   5          Name$ 	PieChart1��5&   +   $��	0,0,-6,41q�6\   $y�Visible=�6o   &    LiveCharts.Wpf.Charts.Base.Chart   Series 6>   �� *
PieSeries1+6o   �� 5)      -   5         ��Name$ 	PieTitle1��5)   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5*      -   5        	 ��Name$	 
Rectangle1��5*   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   
 J� Opacity$	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5+      -   5         ��Name$ WarningIcon1��5+   $   $��/images/warningIcon.png��6:   $��40��6[   $��40��6f   $��Right=�5,   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5/      -   5        $ CriticalIcon1��5/   $   $��/images/criticalIcon.png��6;   $��50��6]   $��50��6h   $��Center=�50   !   $��Center=�6>   $y�	Collapsed=�6Y   52      54      �� 55      $��0��6   $��1��6*   
�������$��1q�6M   $ 10q�6a   ._�56      � .+�57        -   5        $ 	PieChart2��57   +   $��	0,0,-6,41q�6\   $y�Visible=�6o    6>   �� *
PieSeries2+6o   �� 59      -	   5        $ 	PieTitle2��59   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5:      -
   5        $	 
Rectangle2��5:   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5;      -   5        $ WarningIcon2��5;   $   $��/images/warningIcon.png��6:   $��40��6[   $��40��6f   $��Right=�5<   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5?      -   5        $ CriticalIcon2��5?   $   $��/images/criticalIcon.png��6;   $��50��6]   $��50��6h   $��Center=�5@   !   $��Center=�6>   $y�	Collapsed=�6Y   5A      5B      �� 5C      $��0��6   $��2��6*   
�������$��1q�6M   $ 10q�6a   ._�5D      � .+�5E        -   5        $ 	PieChart3��5E   +   $��	0,0,-6,41q�6\   $y�	Collapsed=�6o    6>   �� *
PieSeries3+6o   �� 5G      -   5        $ 	PieTitle3��5G   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5H      -   5        $	 
Rectangle3��5H   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5I      -   5        $ WarningIcon3��5I   $   $��/images/warningIcon.png��6:   $��40��6[   $��40��6f   $��Right=�5J   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5M      -   5        $ CriticalIcon3��5M   $   $��/images/criticalIcon.png��6;   $��50��6]   $��50��6h   $��Center=�5N   !   $��Center=�6>   $y�	Collapsed=�6Y   5O      5P      �� 5Q      $��0��6   $��3��6*   
�������$��1q�6M   $ 10q�6a   ._�5R      � .+�5S        -   5        $ 	PieChart4��5S   +   $��	0,0,-6,41q�6\   $y�	Collapsed=�6o    6>   �� *
PieSeries4+6o   �� 5U      -   5        $ 	PieTitle4��5U   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5V      -   5        $	 
Rectangle4��5V   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5W      -   5        $ WarningIcon4��5W   $   $��/images/warningIcon.png��6:   $��40��6[   $��40��6f   $��Right=�5X   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5[      -   5        $ CriticalIcon4��5[   $   $��/images/criticalIcon.png��6;   $��50��6]   $��50��6h   $��Center=�5\   !   $��Center=�6>   $y�	Collapsed=�6Y   5]      5^      �� 5`      $��1��6   $��0��6*   
�������$��1q�6M   $ 10q�6a   ._�5a      � .+�5b        -   5        $ 	PieChart5��5b   +   $��	0,0,-6,41q�6\   $y�	Collapsed=�6o    6>   �� *
PieSeries5+6o   �� 5d      -   5        $ 	PieTitle5��5d   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5e      -   5        $	 
Rectangle5��5e   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5f      -   5        $ WarningIcon5��5f   $   $��images/warningIcon.png��6:   $��40��6Z   $��40��6e   $��Right=�5g   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5j      -   5        $ CriticalIcon5��5j   $   $��images/criticalIcon.png��6;   $��50��6\   $��50��6g   $��Center=�5k   !   $��Center=�6>   $y�	Collapsed=�6Y   5l      5m      �� 5n      $��1��6   $��1��6*   
�������$��1q�6M   $ 10q�6a   ._�5o      � .+�5p        -   5        $ 	PieChart6��5p   +   $��	0,0,-6,41q�6\   $y�	Collapsed=�6o    6>   �� *
PieSeries6+6o   �� 5r      -   5        $ 	PieTitle6��5r   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5s      -   5        $	 
Rectangle6��5s   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5t      -   5        $ WarningIcon6��5t   $   $��images/warningIcon.png��6:   $��40��6Z   $��40��6e   $��Right=�5u   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5x      -    5        $ CriticalIcon6��5x   $   $��images/criticalIcon.png��6;   $��50��6\   $��50��6g   $��Center=�5y   !   $��Center=�6>   $y�	Collapsed=�6Y   5z      5{      �� 5|      $��1��6   $��2��6*   
�������$��1q�6M   $ 10q�6a   ._�5}      � .+�5~        -!   5        $ 	PieChart7��5~   +   $��	0,0,-6,41q�6\   $y�	Collapsed=�6o    6>   �� *
PieSeries7+6o   �� 5�      -"   5        $ 	PieTitle7��5�   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5�      -#   5        $	 
Rectangle7��5�   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5�      -$   5        $ WarningIcon7��5�   $   $��images/warningIcon.png��6:   $��40��6Z   $��40��6e   $��Right=�5�   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5�      -%   5        $ CriticalIcon7��5�   $   $��images/criticalIcon.png��6;   $��50��6\   $��50��6g   $��Center=�5�   !   $��Center=�6>   $y�	Collapsed=�6Y   5�      5�      �� 5�      $��1��6   $��3��6*   
�������$��1q�6M   $ 10q�6a   ._�5�      � .+�5�        -&   5        $ 	PieChart8��5�   +   $��	0,0,-6,41q�6\   $y�	Collapsed=�6o    6>   �� *
PieSeries8+6o   �� 5�      -'   5        $ 	PieTitle8��5�   (   $�� ��6;   $��
0,187,-6,0q�6C   $��Center=�6W   $��Center=�6t   $��36(�6�   6�   �� 5�      -(   5        $	 
Rectangle8��5�   (   $��Center=�6<   $	��228��6Y   
���   �$��Center=�6u   $	��249��6�   $	
 0.5Q�6�   ���	#FFE42121$y�	Collapsed=�6�   �� 5�      -)   5        $ WarningIcon8��5�   $   $��images/warningIcon.png��6:   $��40��6Z   $��40��6e   $��Right=�5�   !   $	��Top=�6=   $y�	Collapsed=�6U   �� 5�      -*   5        $ CriticalIcon8��5�   $   $��images/criticalIcon.png��6;   $��50��6\   $��50��6g   $��Center=�5�   !   $��Center=�6>   $y�	Collapsed=�6Y   5�      5�      5�      5�      �� 5�       �� Orientation$ Vertical=�6   $��2��60   $��0��6=   $��Center=�6M   $	��Top=�6j   $��10q�6�   .+�5�      �� -+   5         ��Name$ PieChart1Slider��5�       � Minimum$ 0Q�62    � Maximum$	 100Q�6>    � Value$	 100Q�6L    �� TickFrequency$ 10Q�6X    �� IsSnapToTickEnabled$
 True��6k   $	��200��6�   6�   �� 5�      -,   5        $ SliderValueText��5�      $��PieChart1 Value: 100%��65   $��14(�6R   $��Center=�6`   5�      � 5�      $��36,606,36,30q�6   ���	#FFF8EAEA 5�      �� 5�      $��* �6$    5�      �� 5�      $��* �6'   .+�5�      &   System.Windows.Controls.DataGrid  --   5        $��DataGridTable��5�      $��20(�6T   ���	#FFF8EAEA
�������   HorizontalGridLinesBrush
 �����   VerticalGridLinesBrush
 �����$��Stretch=�6�   $��Stretch=�6�   $	��100��6  ��60   �� *ExampleData+5�      5�      5�      5�      5�      