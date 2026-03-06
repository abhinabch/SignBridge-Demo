import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { BookOpen, Search } from 'lucide-react';
import { useState } from 'react';
import {
    ALL_GESTURES,
    ALPHABET_SIGNS,
    COMMON_SIGNS,
    GESTURE_DIFFICULTY,
    GESTURE_DESCRIPTIONS,
} from '../data/gestureLabels';

type FilterType = 'all' | 'alphabet' | 'common' | 'beginner' | 'intermediate';

export function GlossaryGrid() {
    const [filter, setFilter] = useState<FilterType>('all');
    const [search, setSearch] = useState('');

    const filtered = ALL_GESTURES.filter((g) => {
        // Search filter
        if (search && !g.toLowerCase().includes(search.toLowerCase())) return false;
        // Category filter
        switch (filter) {
            case 'alphabet': return ALPHABET_SIGNS.includes(g);
            case 'common': return COMMON_SIGNS.includes(g);
            case 'beginner': return GESTURE_DIFFICULTY[g] === 'Beginner';
            case 'intermediate': return GESTURE_DIFFICULTY[g] === 'Intermediate';
            default: return true;
        }
    });

    const filterButtons: { label: string; value: FilterType }[] = [
        { label: 'All', value: 'all' },
        { label: 'Alphabet', value: 'alphabet' },
        { label: 'Common Signs', value: 'common' },
        { label: 'Beginner', value: 'beginner' },
        { label: 'Intermediate', value: 'intermediate' },
    ];

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <BookOpen className="w-5 h-5 text-blue-600" />
                        Sign Language Glossary
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-600 mb-4">
                        Browse all {ALL_GESTURES.length} supported signs. Each card shows the hand configuration needed.
                    </p>

                    {/* Search */}
                    <div className="relative mb-4">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search signs..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            className="w-full pl-9 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                    </div>

                    {/* Filters */}
                    <div className="flex flex-wrap gap-2 mb-4">
                        {filterButtons.map(({ label, value }) => (
                            <button
                                key={value}
                                onClick={() => setFilter(value)}
                                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${filter === value
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                            >
                                {label}
                            </button>
                        ))}
                    </div>

                    <div className="text-xs text-gray-500 mb-3">{filtered.length} signs</div>
                </CardContent>
            </Card>

            {/* Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {filtered.map((gesture) => {
                    const difficulty = GESTURE_DIFFICULTY[gesture] || 'Beginner';
                    const desc = GESTURE_DESCRIPTIONS[gesture] || 'Perform the sign';
                    const isAlphabet = ALPHABET_SIGNS.includes(gesture);

                    return (
                        <Card key={gesture} className="hover:shadow-md transition-shadow">
                            <CardContent className="p-4">
                                <div className="flex items-start justify-between mb-2">
                                    <div className={`text-xl font-bold ${isAlphabet ? 'text-indigo-600' : 'text-blue-600'}`}>
                                        {gesture}
                                    </div>
                                    <Badge
                                        variant="secondary"
                                        className={`text-xs ${difficulty === 'Beginner'
                                                ? 'bg-green-100 text-green-700'
                                                : 'bg-orange-100 text-orange-700'
                                            }`}
                                    >
                                        {difficulty}
                                    </Badge>
                                </div>
                                <p className="text-xs text-gray-600 leading-relaxed">
                                    {desc}
                                </p>
                            </CardContent>
                        </Card>
                    );
                })}
            </div>

            {filtered.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                    <p className="text-sm">No signs match your search.</p>
                </div>
            )}
        </div>
    );
}
