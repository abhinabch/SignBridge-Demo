import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { BookOpen, PlayCircle, CheckCircle, Lock, Star, TrendingUp } from 'lucide-react';
import { useState } from 'react';

const lessons = [
  {
    id: 1,
    title: 'Basic Greetings',
    description: 'Learn common greetings in ASL',
    signs: ['Hello', 'Goodbye', 'Nice to meet you'],
    difficulty: 'Beginner',
    duration: '5 min',
    completed: true,
    progress: 100
  },
  {
    id: 2,
    title: 'Common Phrases',
    description: 'Essential everyday phrases',
    signs: ['Thank you', 'Please', 'Sorry', 'Help'],
    difficulty: 'Beginner',
    duration: '8 min',
    completed: true,
    progress: 100
  },
  {
    id: 3,
    title: 'Numbers 1-10',
    description: 'Count from 1 to 10 in sign language',
    signs: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    difficulty: 'Beginner',
    duration: '10 min',
    completed: false,
    progress: 60
  },
  {
    id: 4,
    title: 'Family Members',
    description: 'Signs for family relationships',
    signs: ['Mother', 'Father', 'Sister', 'Brother'],
    difficulty: 'Beginner',
    duration: '7 min',
    completed: false,
    progress: 0
  },
  {
    id: 5,
    title: 'Emotions & Feelings',
    description: 'Express how you feel',
    signs: ['Happy', 'Sad', 'Angry', 'Excited'],
    difficulty: 'Intermediate',
    duration: '12 min',
    completed: false,
    progress: 0
  },
  {
    id: 6,
    title: 'Question Words',
    description: 'Learn to ask questions',
    signs: ['What', 'Where', 'When', 'Why', 'How'],
    difficulty: 'Intermediate',
    duration: '15 min',
    completed: false,
    progress: 0,
    locked: true
  }
];

const categories = [
  { name: 'Alphabet', icon: '🔤', count: 26 },
  { name: 'Numbers', icon: '🔢', count: 10 },
  { name: 'Colors', icon: '🎨', count: 12 },
  { name: 'Food', icon: '🍎', count: 20 },
  { name: 'Animals', icon: '🐕', count: 15 },
  { name: 'Weather', icon: '☀️', count: 8 }
];

export function LearnSection() {
  const [selectedLesson, setSelectedLesson] = useState<number | null>(null);
  
  const completedLessons = lessons.filter(l => l.completed).length;
  const totalLessons = lessons.length;
  const overallProgress = (completedLessons / totalLessons) * 100;

  return (
    <div className="space-y-6">
      {/* Progress Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            Your Learning Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Overall Progress</span>
              <span className="text-sm">{completedLessons} of {totalLessons} lessons</span>
            </div>
            <Progress value={overallProgress} className="h-3" />
            
            <div className="grid grid-cols-3 gap-4 mt-6">
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <div className="text-2xl text-blue-600">{completedLessons}</div>
                <div className="text-xs text-gray-600 mt-1">Completed</div>
              </div>
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <div className="text-2xl text-green-600">45</div>
                <div className="text-xs text-gray-600 mt-1">Signs Learned</div>
              </div>
              <div className="text-center p-3 bg-purple-50 rounded-lg">
                <div className="text-2xl text-purple-600">12</div>
                <div className="text-xs text-gray-600 mt-1">Day Streak</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Categories */}
      <Card>
        <CardHeader>
          <CardTitle>Browse by Category</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3">
            {categories.map((category) => (
              <button
                key={category.name}
                className="p-4 bg-gradient-to-br from-gray-50 to-gray-100 hover:from-blue-50 hover:to-blue-100 rounded-lg transition-colors text-center"
              >
                <div className="text-3xl mb-2">{category.icon}</div>
                <div className="text-xs mb-1">{category.name}</div>
                <div className="text-xs text-gray-500">{category.count} signs</div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Lessons */}
      <Card>
        <CardHeader>
          <CardTitle>Learning Path</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {lessons.map((lesson) => (
              <div
                key={lesson.id}
                className={`p-4 border-2 rounded-lg transition-all ${
                  lesson.locked
                    ? 'bg-gray-50 border-gray-200 opacity-60'
                    : 'bg-white border-gray-200 hover:border-blue-400 cursor-pointer'
                }`}
                onClick={() => !lesson.locked && setSelectedLesson(lesson.id)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-sm">{lesson.title}</h3>
                      {lesson.completed && (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      )}
                      {lesson.locked && (
                        <Lock className="w-4 h-4 text-gray-400" />
                      )}
                    </div>
                    <p className="text-xs text-gray-600 mb-2">{lesson.description}</p>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        {lesson.difficulty}
                      </Badge>
                      <span className="text-xs text-gray-500">{lesson.duration}</span>
                    </div>
                  </div>
                  
                  {!lesson.locked && (
                    <Button size="sm" variant={lesson.completed ? "outline" : "default"}>
                      {lesson.completed ? (
                        <>Review</>
                      ) : lesson.progress > 0 ? (
                        <>Continue</>
                      ) : (
                        <>
                          <PlayCircle className="w-4 h-4 mr-1" />
                          Start
                        </>
                      )}
                    </Button>
                  )}
                </div>

                {lesson.progress > 0 && !lesson.locked && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-600">
                      <span>Progress</span>
                      <span>{lesson.progress}%</span>
                    </div>
                    <Progress value={lesson.progress} className="h-2" />
                  </div>
                )}

                {lesson.locked && (
                  <p className="text-xs text-gray-500 mt-2">
                    Complete previous lessons to unlock
                  </p>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Practice Reminder */}
      <Card className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-0">
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <div className="bg-white/20 p-3 rounded-full">
              <Star className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3 className="mb-1">Daily Practice Goal</h3>
              <p className="text-sm text-blue-100 mb-3">
                Practice for 10 minutes daily to maintain your streak!
              </p>
              <Button size="sm" className="bg-white text-blue-600 hover:bg-blue-50">
                Start Daily Practice
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
