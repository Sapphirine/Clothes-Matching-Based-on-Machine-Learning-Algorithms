%Because some items only have photos but don't have keywords,we can't
%extract features of these items, thus, we need to remove these items out
%of our dataset
%Besides,we can't ensure every item in the fashion experts' collocation are
%also in the items.txt. Thus, they may not have features and photos, we
%also need to remove these items out. Because these items without photos
%can't be shown in the result.
%Import items.txt(all items information,including id, catagory, feature)
%Import match.txt(the collocation from fashion experts, including
%match_map_id and items_id)
new_match=match(:,1)
for i=2:20
    new_match=[new_match;match(:,i)];
end
 new_match(isnan(new_match(:,1))==1)=[];
 %get the id of common items of items.txt and new_match
 comm = intersect(new_match,items);
 items_feature=items(ismember(items(:,1),comm),:);
 %get items(include their features) that has keywords and photos
 

     