import { HomePage, ReactNativeMLKitPage, FastTFLitePage, DemoTFLitePage } from '../pages/index';
//  ReactNativeMLKitPage
import { NavigationContainer } from '@react-navigation/native';
import type { BottomTabNavigationOptions } from '@react-navigation/bottom-tabs';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { MaterialDesignIcons, MaterialDesignIconsIconName } from '@react-native-vector-icons/material-design-icons';
type TabParamList = {
  Home: undefined;
  MLKitPage: undefined;
  TFLite?: undefined;
};
const Tab = createBottomTabNavigator();

export function getTabScreenOptions(routeName: string): BottomTabNavigationOptions {
  return {
    tabBarIcon: ({ focused, color, size }) => {
      let iconName: MaterialDesignIconsIconName;

      switch (routeName) {
        case 'Home':
          iconName = 'home-outline';
          break;
        case 'MLKitPage':
          iconName = 'text-recognition';
          break;
        case 'FastTFLitePage':
          iconName = 'scan-helper';
          break;
        case 'DemoTFLitePage':
          iconName = 'line-scan';
          break;
        default:
          iconName = 'circle';
      }

      return <MaterialDesignIcons name={iconName} size={size} color={color} />;
    },
    tabBarActiveTintColor: '#007AFF',
    tabBarInactiveTintColor: 'gray',
  };
}
export function CustomBottomTabNavigator() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }): BottomTabNavigationOptions => getTabScreenOptions(route.name)}
        initialRouteName="Home"
      >
        <Tab.Screen name="Home" component={HomePage} />
        <Tab.Screen name="MLKitPage" component={ReactNativeMLKitPage} />
        <Tab.Screen name="FastTFLitePage" component={FastTFLitePage} />
        <Tab.Screen name="DemoTFLitePage" component={DemoTFLitePage} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
