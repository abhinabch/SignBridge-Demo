import { Camera, Settings, History, BookOpen, User, Home, Scan, X, AlertCircle, ArrowLeft } from "lucide-react";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { useState, useRef, useEffect } from "react";
import { Alert, AlertDescription } from "./components/ui/alert";
import { SignLanguageRecognizer } from "./components/SignLanguageRecognizer";
import { LearnSection } from "./components/LearnSection";
import { ProfileSection } from "./components/ProfileSection";
import { SettingsSection } from "./components/SettingsSection";

type Section = 'home' | 'camera' | 'learn' | 'profile' | 'settings';

export default function App() {
  const [currentSection, setCurrentSection] = useState<Section>('home');

  // Render different sections based on current selection
  if (currentSection !== 'home') {
    const sectionTitles = {
      camera: 'Sign Recognition',
      learn: 'Learn Sign Language',
      profile: 'Your Profile',
      settings: 'Settings'
    };

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <Button 
              variant="ghost" 
              onClick={() => setCurrentSection('home')}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Home
            </Button>
            <h1 className="text-2xl text-gray-800">
              {sectionTitles[currentSection as keyof typeof sectionTitles]}
            </h1>
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => setCurrentSection('settings')}
            >
              <Settings className="w-5 h-5" />
            </Button>
          </div>

          {/* Section Content */}
          {currentSection === 'camera' && <SignLanguageRecognizer />}
          {currentSection === 'learn' && <LearnSection />}
          {currentSection === 'profile' && <ProfileSection />}
          {currentSection === 'settings' && <SettingsSection />}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4 py-8">
      {/* Mobile Phone Mockup */}
      <div className="w-full max-w-sm h-[90vh] flex flex-col">
        {/* Phone Frame */}
        <div className="bg-gray-900 rounded-[3rem] p-3 shadow-2xl flex flex-col flex-1 overflow-hidden">
          {/* Phone Screen */}
          <div className="bg-white rounded-[2.5rem] overflow-hidden flex flex-col flex-1">
            {/* Status Bar */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3 flex items-center justify-between text-white text-xs flex-shrink-0">
              <span>9:41</span>
              <div className="flex items-center gap-1">
                <div className="w-4 h-3 border border-white rounded-sm"></div>
                <div className="w-1 h-3 bg-white rounded-sm"></div>
              </div>
            </div>

            {/* App Header */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 pt-4 pb-8 flex-shrink-0">
              <div className="flex items-center justify-between mb-4">
                {/* Logo and Title */}
                <div className="flex items-center gap-3">
                  {/* SignBridge Logo */}
                  <svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    {/* Orange Person */}
                    <circle cx="35" cy="20" r="8" fill="#FFA500"/>
                    <path d="M35 30L25 45V60M35 30L45 45V60M35 35L20 50M35 35L50 50" stroke="#FFA500" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    
                    {/* Blue Person */}
                    <circle cx="65" cy="20" r="8" fill="#00A8E8"/>
                    <path d="M65 30L55 45V60M65 30L75 45V60M65 35L50 50M65 35L80 50" stroke="#00A8E8" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    
                    {/* Connecting Bridge */}
                    <line x1="35" y1="50" x2="65" y2="50" stroke="#FFA500" strokeWidth="3" strokeLinecap="round"/>
                    <circle cx="35" cy="50" r="4" fill="#FFA500"/>
                    <circle cx="65" cy="50" r="4" fill="#00A8E8"/>
                  </svg>
                  <div>
                    <h1 className="text-white text-2xl font-bold">SignBridge</h1>
                    <p className="text-blue-100 text-xs">Breaking barriers</p>
                  </div>
                </div>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-white hover:bg-white/20"
                  onClick={() => setCurrentSection('settings')}
                >
                  <Settings className="w-5 h-5" />
                </Button>
              </div>
              <p className="text-blue-100 text-sm">Real-time sign language translation</p>
            </div>

            {/* Main Content - Scrollable */}
            <div className="flex-1 overflow-y-auto px-6 pt-4 pb-4">
              {/* Camera Card - Primary Action */}
              <Card className="p-6 mb-4 bg-gradient-to-br from-indigo-500 to-purple-600 border-0 shadow-lg">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h2 className="text-white mb-1">Real-Time Translation</h2>
                    <p className="text-indigo-100 text-sm">Start camera to translate signs</p>
                  </div>
                  <div className="bg-white/20 p-4 rounded-full backdrop-blur-sm">
                    <Camera className="w-8 h-8 text-white" />
                  </div>
                </div>
                <Button 
                  className="w-full mt-4 bg-white text-indigo-600 hover:bg-indigo-50"
                  onClick={() => setCurrentSection('camera')}
                >
                  <Scan className="w-4 h-4 mr-2" />
                  Start Camera
                </Button>
              </Card>

              {/* Quick Actions Grid */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <Card 
                  className="p-4 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => setCurrentSection('learn')}
                >
                  <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mb-3">
                    <History className="w-6 h-6 text-blue-600" />
                  </div>
                  <h3 className="text-sm mb-1">History</h3>
                  <p className="text-gray-500 text-xs">View past translations</p>
                </Card>

                <Card 
                  className="p-4 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => setCurrentSection('learn')}
                >
                  <div className="bg-green-100 w-12 h-12 rounded-full flex items-center justify-center mb-3">
                    <BookOpen className="w-6 h-6 text-green-600" />
                  </div>
                  <h3 className="text-sm mb-1">Learn</h3>
                  <p className="text-gray-500 text-xs">Sign language guide</p>
                </Card>
              </div>

              {/* Recent Activity */}
              <div className="mb-4">
                <h3 className="text-sm text-gray-600 mb-3">Recent Translations</h3>
                <Card className="p-4 mb-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm mb-1">Hello</p>
                      <p className="text-xs text-gray-500">2 hours ago</p>
                    </div>
                    <div className="text-xs text-blue-600">ASL</div>
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm mb-1">Thank you</p>
                      <p className="text-xs text-gray-500">Yesterday</p>
                    </div>
                    <div className="text-xs text-blue-600">ASL</div>
                  </div>
                </Card>
              </div>
            </div>

            {/* Bottom Navigation - Fixed to bottom */}
            <div className="bg-white border-t border-gray-200 px-6 py-3 flex items-center justify-around flex-shrink-0">
              <Button 
                variant="ghost" 
                size="icon" 
                className="flex-col h-auto py-2 text-blue-600"
                onClick={() => setCurrentSection('home')}
              >
                <Home className="w-5 h-5 mb-1" />
                <span className="text-xs">Home</span>
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="flex-col h-auto py-2 text-gray-400"
                onClick={() => setCurrentSection('camera')}
              >
                <Camera className="w-5 h-5 mb-1" />
                <span className="text-xs">Camera</span>
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="flex-col h-auto py-2 text-gray-400"
                onClick={() => setCurrentSection('learn')}
              >
                <BookOpen className="w-5 h-5 mb-1" />
                <span className="text-xs">Learn</span>
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="flex-col h-auto py-2 text-gray-400"
                onClick={() => setCurrentSection('profile')}
              >
                <User className="w-5 h-5 mb-1" />
                <span className="text-xs">Profile</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}