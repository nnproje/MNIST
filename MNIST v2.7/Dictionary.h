#pragma once
#ifndef DICTIONARY_HEADER
#define DICTIONARY_HEADER
#include <iostream>
#include <string>
#include <process.h>
#include <conio.h>
#include <list>
#include <vector>
#include <stdlib.h>
#include <cstring>
#include "VectVolume.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////*ENTRY*//////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
class Entry {
public:
	K KEY;															//key of entry
	V VALUE;														//value of entry
public:
	Entry(K k= K() ,const V& v= V()) : KEY(k), VALUE(v) { }			//constructor
	const K& key() const { return KEY; }							//returns const reference to KEY
	const V& value() const { return VALUE; }						//returns const reference to VALUE
	void setKey(K k) { KEY = k; }									//sets KEY to k
	void setValue(V v) { VALUE = v; }								//sets VALUE to v
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////*Dictionary*//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
class Dictionary
{
private:
	typedef Entry<K, V> ENTRY;
	typedef std::list<ENTRY> Bucket;
	typedef typename Bucket::iterator BItor;
private:
	Bucket bkt;						//list of entries
	int SIZE;						//size of dictionary
	std::string NAME;				//name of dictionary
public:
	Dictionary(std::string name= "YOU DIDN'T GIVE IT A NAME!"); //constructor
	int  size();										        //return the size of dictionary
	bool empty();										        //return true if the dictionary is empty
	bool exist(K key);										    //return true if key exists in the dictionary, else it prints err msg
	void put(K key, V& value);								    //put ENTRY(key,value) into dictionary, if the key already exists it prints err msg
	void erase(K key);										    //removes the entry with key, if key does not exist it prints err msg
	void DeleteThenErase(K key);								//deletes allocated element then removes the entry with key, if key does not exist it prints err msg
	void DeleteThenEraseObj(K key);								//deletes allocated element then removes the entry with key, if key does not exist it prints err msg
	void replace(K key,V value);                                //replaces the value of the entry with the provided key, if the key does not exist it prints err msg
	void DeleteThenReplace(K key, V value);                     //replaces the value of the entry with the provided key, if the key does not exist it prints err msg
	void DeleteThenReplaceObj(K key, V value);                  //replaces the value of the entry with the provided key, if the key does not exist it prints err msg
	void clear();												//removes all entries in the dictionary
	void DeleteThenClear();										//deletes allocated elements then removes all entries in the dictionary (Matrix)
	void DeleteThenClearObj();								    //deletes allocated elements then removes all entries in the dictionary (Matrix)
	void print();											    //print all entries in the dictionary (assuming that the key could be directly printed and the value has its own print function)
	void printObj();											//print all entries in the dictionary (assuming that the key could be directly printed and the value has its own print function)
	void setName(std::string name);                             //sets the dictionary name
	BItor find(K key);										    //returns an iterator to the entry with key, if key does not exist it prints err msg
	const V& operator[](K key);								    //returns a const refrence the the value of the entry with key
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
Dictionary<K,V>::Dictionary(std::string name) : SIZE(0), NAME(name)
{}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
int Dictionary<K, V>::size()
{
	return bkt.size();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
bool Dictionary<K, V>::empty()
{
	return size() == 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
bool Dictionary<K, V>::exist(K key)
{
	for (BItor i = bkt.begin(); i != bkt.end(); ++i)
	{
		if(i->KEY == key)
			return true;
	}
	return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
typename Dictionary<K,V>::BItor Dictionary<K, V>::find(K key)
{
	for (BItor i = bkt.begin(); i != bkt.end(); ++i)
	{
		if(i->KEY == key)
			return i;
	}
	std::cout << "This key does not exist!" << std::endl;
	return bkt.end();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K,V>::put(K key, V& value)
{
	if (exist(key))
	{
		std::cout << "This key "<< key << " is used before!" << std::endl;
	}
	else
	{
		bkt.push_front(ENTRY(key, value));
		SIZE++;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K,V>::replace(K key,V value)
{
    if (!exist(key))
		std::cout << "This key does not exist!" << std::endl;
    else
    {
        BItor i = find(key);
        i->setValue(value);
    }

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::DeleteThenReplace(K key, V value)
{
	if (!exist(key))
		std::cout << "This key does not exist!" << std::endl;
	else
	{
		BItor i = find(key);
		delete (i->VALUE);
		i->setValue(value);
	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::DeleteThenReplaceObj(K key, V value)
{
	if (!exist(key))
		std::cout << "This key does not exist!" << std::endl;
	else
	{
		BItor i = find(key);
		(i->VALUE).DELETE();
		i->setValue(value);
	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::erase(K key)
{
	if (!exist(key))
		std::cout << "This key does not exist!" << std::endl;
	else
	{
		BItor i = find(key);
		bkt.erase(i);
		SIZE--;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::DeleteThenErase(K key)
{
	if (!exist(key))
		std::cout << "This key does not exist!" << std::endl;
	else
	{
		BItor i = find(key);
		delete (i->VALUE);
		bkt.erase(i);
		SIZE--;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::DeleteThenEraseObj(K key)
{
	if (!exist(key))
		std::cout << "This key does not exist!" << std::endl;
	else
	{
		BItor i = find(key);
		(i->VALUE).DELETE();
		bkt.erase(i);
		SIZE--;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::clear()
{
	bkt.clear();
	SIZE = 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::DeleteThenClear()
{
	for (BItor i = bkt.begin(); i != bkt.end(); ++i)
	{
		delete (i->VALUE);
	}
	bkt.clear();
	SIZE = 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::DeleteThenClearObj()
{
	for (BItor i = bkt.begin(); i != bkt.end(); ++i)
	{
		(i->VALUE).DELETE();
	}
	bkt.clear();
	SIZE = 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
const V& Dictionary<K, V>::operator[](K key)
{
	if (exist(key))
	{
		BItor i = find(key);
		return i->value();
	}
	std::cout << "This key does not exist!" << std::endl << std::endl;
	std::cout << "Press enter to end.." << std::endl;
	_getche();
	exit(0);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::print()
{
	std::cout << std::endl << "The size of " << NAME << " : " << SIZE << std::endl;
	std::cout << std::endl << "The contents of " << NAME << " :" << std::endl << std::endl;
	for (BItor i = bkt.begin(); i != bkt.end(); ++i)
	{
		std::cout << i->key() << std::endl;
		_getche();
		i->value()->print();
		std::cout << std::endl<<std::endl<<std::endl;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::printObj()
{
	std::cout << std::endl << "The size of " << NAME << " : " << SIZE << std::endl;
	std::cout << std::endl << "The contents of " << NAME << " :" << std::endl << std::endl;
	for (BItor i = bkt.begin(); i != bkt.end(); ++i)
	{
		std::cout << i->key() << std::endl;
		_getche();
		VectVolume p = i->value();
		p.print();
		std::cout << std::endl<<std::endl<<std::endl;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename K, typename V>
void Dictionary<K, V>::setName(std::string name)
{
    NAME=name;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // !DICTIONARY_HEADER



/*the problem is vectvolume is a pointer, when passing it a new obeject is created that have the same pointer, when the new object is deleted the whole element*/
