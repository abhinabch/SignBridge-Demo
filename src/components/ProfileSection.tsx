import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { 
  User, 
  Award, 
  Calendar, 
  TrendingUp, 
  Target,
  Settings,
  LogOut,
  Edit,
  Trophy,
  Flame,
  Star
} from 'lucide-react';

const achievements = [
  { id: 1, title: 'First Steps', description: 'Complete your first lesson', icon: '🎯', earned: true },
  { id: 2, title: 'Week Warrior', description: '7-day learning streak', icon: '🔥', earned: true },
  { id: 3, title: 'Sign Master', description: 'Learn 50 signs', icon: '⭐', earned: true },
  { id: 4, title: 'Perfect Practice', description: '100% accuracy on a lesson', icon: '💯', earned: false },
  { id: 5, title: 'Monthly Milestone', description: '30-day streak', icon: '🏆', earned: false },
  { id: 6, title: 'Community Helper', description: 'Help others learn', icon: '🤝', earned: false }
];

const recentActivity = [
  { action: 'Completed lesson', detail: 'Common Phrases', time: '2 hours ago', icon: '✓' },
  { action: 'Practiced signs', detail: '15 signs reviewed', time: 'Yesterday', icon: '📚' },
  { action: 'Earned achievement', detail: 'Week Warrior', time: '2 days ago', icon: '🏆' },
  { action: 'Translation session', detail: '10 min practice', time: '3 days ago', icon: '📹' }
];

const stats = [
  { label: 'Total Signs Learned', value: '45', icon: Award, color: 'text-blue-600', bg: 'bg-blue-50' },
  { label: 'Current Streak', value: '12 days', icon: Flame, color: 'text-orange-600', bg: 'bg-orange-50' },
  { label: 'Lessons Completed', value: '8', icon: Trophy, color: 'text-green-600', bg: 'bg-green-50' },
  { label: 'Practice Hours', value: '4.5h', icon: TrendingUp, color: 'text-purple-600', bg: 'bg-purple-50' }
];

export function ProfileSection() {
  return (
    <div className="space-y-6">
      {/* Profile Header */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <Avatar className="w-20 h-20">
              <AvatarImage src="" />
              <AvatarFallback className="bg-gradient-to-br from-blue-600 to-indigo-600 text-white text-xl">
                JD
              </AvatarFallback>
            </Avatar>
            
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-xl">John Doe</h2>
                <Button size="sm" variant="outline">
                  <Edit className="w-4 h-4 mr-2" />
                  Edit
                </Button>
              </div>
              <p className="text-sm text-gray-600 mb-3">Learning ASL since November 2024</p>
              
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="flex items-center gap-1">
                  <Star className="w-3 h-3" />
                  Level 2
                </Badge>
                <Badge variant="outline">Beginner</Badge>
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-4 gap-2 mt-6 pt-6 border-t">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="text-center">
                  <div className={`${stat.bg} ${stat.color} w-10 h-10 rounded-full flex items-center justify-center mx-auto mb-2`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="text-sm mb-0.5">{stat.value}</div>
                  <div className="text-xs text-gray-500">{stat.label.split(' ')[0]}</div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Learning Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            Learning Statistics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className={`${stat.bg} ${stat.color} w-10 h-10 rounded-full flex items-center justify-center`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-sm">{stat.label}</span>
                  </div>
                  <span className={`${stat.color}`}>{stat.value}</span>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Achievements */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy className="w-5 h-5 text-yellow-600" />
            Achievements
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3">
            {achievements.map((achievement) => (
              <div
                key={achievement.id}
                className={`p-4 rounded-lg border-2 ${
                  achievement.earned
                    ? 'bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-200'
                    : 'bg-gray-50 border-gray-200 opacity-60'
                }`}
              >
                <div className="text-3xl mb-2">{achievement.icon}</div>
                <h4 className="text-sm mb-1">{achievement.title}</h4>
                <p className="text-xs text-gray-600">{achievement.description}</p>
                {achievement.earned && (
                  <Badge variant="secondary" className="mt-2 text-xs">
                    Earned
                  </Badge>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-600" />
            Recent Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentActivity.map((activity, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 text-sm">
                  {activity.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm mb-0.5">{activity.action}</div>
                  <div className="text-xs text-gray-600">{activity.detail}</div>
                </div>
                <div className="text-xs text-gray-500 flex-shrink-0">{activity.time}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Goals */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-green-600" />
            Current Goals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-green-50 border-2 border-green-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Reach 30-day streak</span>
                <span className="text-sm text-green-600">12/30 days</span>
              </div>
              <div className="w-full bg-green-200 rounded-full h-2">
                <div className="bg-green-600 h-2 rounded-full" style={{ width: '40%' }}></div>
              </div>
            </div>

            <div className="p-4 bg-blue-50 border-2 border-blue-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Learn 100 signs</span>
                <span className="text-sm text-blue-600">45/100 signs</span>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: '45%' }}></div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Account Actions */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-2">
            <Button variant="outline" className="w-full justify-start">
              <Settings className="w-4 h-4 mr-2" />
              Account Settings
            </Button>
            <Button variant="outline" className="w-full justify-start text-red-600 hover:text-red-700 hover:bg-red-50">
              <LogOut className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
