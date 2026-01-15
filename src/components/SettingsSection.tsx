import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { 
  Settings,
  Bell,
  Camera,
  Volume2,
  Moon,
  Globe,
  Shield,
  HelpCircle,
  Info,
  Trash2,
  Download,
  Smartphone,
  Wifi,
  Zap
} from 'lucide-react';
import { useState } from 'react';

export function SettingsSection() {
  const [notifications, setNotifications] = useState(true);
  const [soundEffects, setSoundEffects] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [autoStart, setAutoStart] = useState(false);
  const [hapticFeedback, setHapticFeedback] = useState(true);
  const [dataCollection, setDataCollection] = useState(false);

  return (
    <div className="space-y-6">
      {/* App Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-blue-600" />
            General Settings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b">
              <div className="flex items-center gap-3">
                <Bell className="w-5 h-5 text-gray-600" />
                <div>
                  <Label htmlFor="notifications" className="text-sm cursor-pointer">
                    Push Notifications
                  </Label>
                  <p className="text-xs text-gray-500">Get reminders to practice</p>
                </div>
              </div>
              <Switch
                id="notifications"
                checked={notifications}
                onCheckedChange={setNotifications}
              />
            </div>

            <div className="flex items-center justify-between py-3 border-b">
              <div className="flex items-center gap-3">
                <Volume2 className="w-5 h-5 text-gray-600" />
                <div>
                  <Label htmlFor="sound" className="text-sm cursor-pointer">
                    Sound Effects
                  </Label>
                  <p className="text-xs text-gray-500">Play sounds for actions</p>
                </div>
              </div>
              <Switch
                id="sound"
                checked={soundEffects}
                onCheckedChange={setSoundEffects}
              />
            </div>

            <div className="flex items-center justify-between py-3 border-b">
              <div className="flex items-center gap-3">
                <Moon className="w-5 h-5 text-gray-600" />
                <div>
                  <Label htmlFor="darkMode" className="text-sm cursor-pointer">
                    Dark Mode
                  </Label>
                  <p className="text-xs text-gray-500">Switch to dark theme</p>
                </div>
              </div>
              <Switch
                id="darkMode"
                checked={darkMode}
                onCheckedChange={setDarkMode}
              />
            </div>

            <div className="flex items-center justify-between py-3">
              <div className="flex items-center gap-3">
                <Smartphone className="w-5 h-5 text-gray-600" />
                <div>
                  <Label htmlFor="haptic" className="text-sm cursor-pointer">
                    Haptic Feedback
                  </Label>
                  <p className="text-xs text-gray-500">Vibrate on interactions</p>
                </div>
              </div>
              <Switch
                id="haptic"
                checked={hapticFeedback}
                onCheckedChange={setHapticFeedback}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Camera Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-blue-600" />
            Camera & Recognition
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b">
              <div className="flex items-center gap-3">
                <Zap className="w-5 h-5 text-gray-600" />
                <div>
                  <Label htmlFor="autoStart" className="text-sm cursor-pointer">
                    Auto-Start Recognition
                  </Label>
                  <p className="text-xs text-gray-500">Start detecting when camera opens</p>
                </div>
              </div>
              <Switch
                id="autoStart"
                checked={autoStart}
                onCheckedChange={setAutoStart}
              />
            </div>

            <div className="py-3 border-b">
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm">Camera Quality</Label>
                <Badge variant="secondary">High</Badge>
              </div>
              <p className="text-xs text-gray-500">Higher quality uses more data</p>
            </div>

            <div className="py-3">
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm">Recognition Sensitivity</Label>
                <Badge variant="secondary">Medium</Badge>
              </div>
              <p className="text-xs text-gray-500">Adjust detection threshold</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Language & Region */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="w-5 h-5 text-blue-600" />
            Language & Region
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="py-3 border-b">
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm">App Language</Label>
                <Badge variant="secondary">English</Badge>
              </div>
              <p className="text-xs text-gray-500">Change interface language</p>
            </div>

            <div className="py-3">
              <div className="flex items-center justify-between mb-2">
                <Label className="text-sm">Sign Language</Label>
                <Badge variant="secondary">ASL (American)</Badge>
              </div>
              <p className="text-xs text-gray-500">Choose sign language variant</p>
              <div className="flex flex-wrap gap-2 mt-3">
                <Badge variant="outline" className="cursor-pointer">ASL</Badge>
                <Badge variant="outline" className="cursor-pointer">BSL</Badge>
                <Badge variant="outline" className="cursor-pointer">ISL</Badge>
                <Badge variant="outline" className="cursor-pointer">More...</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Privacy & Security */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-blue-600" />
            Privacy & Security
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b">
              <div className="flex items-center gap-3">
                <Wifi className="w-5 h-5 text-gray-600" />
                <div>
                  <Label htmlFor="dataCollection" className="text-sm cursor-pointer">
                    Anonymous Analytics
                  </Label>
                  <p className="text-xs text-gray-500">Help improve the app</p>
                </div>
              </div>
              <Switch
                id="dataCollection"
                checked={dataCollection}
                onCheckedChange={setDataCollection}
              />
            </div>

            <div className="py-3 border-b">
              <Button variant="outline" size="sm" className="w-full justify-start">
                <Shield className="w-4 h-4 mr-2" />
                Privacy Policy
              </Button>
            </div>

            <div className="py-3">
              <Button variant="outline" size="sm" className="w-full justify-start">
                <Download className="w-4 h-4 mr-2" />
                Download My Data
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data & Storage */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Smartphone className="w-5 h-5 text-blue-600" />
            Data & Storage
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Storage Used</span>
                <span className="text-sm">24.5 MB</span>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: '25%' }}></div>
              </div>
              <p className="text-xs text-gray-600 mt-2">Cache and offline content</p>
            </div>

            <Button variant="outline" size="sm" className="w-full justify-start">
              <Trash2 className="w-4 h-4 mr-2" />
              Clear Cache
            </Button>

            <Button variant="outline" size="sm" className="w-full justify-start text-red-600 hover:text-red-700 hover:bg-red-50">
              <Trash2 className="w-4 h-4 mr-2" />
              Delete All Data
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* About & Support */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-600" />
            About & Support
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <Button variant="outline" size="sm" className="w-full justify-start">
              <HelpCircle className="w-4 h-4 mr-2" />
              Help Center
            </Button>

            <Button variant="outline" size="sm" className="w-full justify-start">
              <Info className="w-4 h-4 mr-2" />
              About SignBridge
            </Button>

            <div className="pt-4 border-t text-center">
              <p className="text-xs text-gray-500">Version 1.0.0</p>
              <p className="text-xs text-gray-500 mt-1">© 2024 SignBridge. All rights reserved.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
